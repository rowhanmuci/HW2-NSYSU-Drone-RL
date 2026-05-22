# Task A: Precision Hovering — MDP Design Notes

## 任務描述

訓練無人機懸停在指定座標 (x, y, z)，並能在 `motionDriftNoise` 擾動（風力模擬）下維持穩定，不發散。

---

## MDP 設計

### 觀測空間 (10-D)

| Index | 意義 | 單位 |
|-------|------|------|
| 0–2   | 無人機位置 (x, y, z) | m |
| 3–5   | 無人機速度 (vx, vy, vz) | m/s |
| 6–8   | 懸停目標 (tx, ty, tz) | m |
| 9     | 到目標的距離 | m |

相較 Task E 的 13-D，移除了 `progress_ratio` 和 `XY unit direction`，因為這兩項對單點懸停沒有意義。

### 動作空間 (3-D)

速度指令 (vx, vy, vz) ∈ [−1, 1] m/s，送至 `/simple_drone/cmd_vel`。

### 獎勵函數（最終版）

```
shaping = (prev_dist − dist) × 100.0  # 強化 shaping（舊版 × 10，詳見下方分析）

r_t = −dist × 0.5                      # 輕微距離懲罰（舊版 × 2.0，降低以避免 baseline 壓制 shaping）
    + shaping                           # 強導航梯度：往目標 0.05 m → +5.0
    − vel_mag × 0.05                    # 速度懲罰（避免震盪）
    − 0.01                              # 時間懲罰（鼓勵效率）

進入成功圈（dist < 0.3 m）：+3.0/step（zone bonus）
成功：連續 N 步 dist < 0.3 m → +100，episode 終止（N 隨 phase 變化）
墜機：z < 0.1 m（step > 20）→ −100，終止
出界：|pos| > 6 m → −50，終止
```

#### 為什麼需要 Potential-Based Shaping？

初版只用 `−dist × 2.0`，理論上有梯度，但在實際訓練中（特別是 Gazebo）導致 policy 完全不移動。

**根本原因：零 advantage 問題**

當所有 episode 的 return 都是均一負值（例如每步 ≈ −4，300 步 = −1200），value function 預測 V(s) ≈ −1200 for 所有狀態，則：

```
Advantage(a|s) = Q(s,a) − V(s) ≈ 0  for all a
```

advantage 接近零 → policy gradient ≈ 0 → policy 停止更新。無人機學會「懸停在起始點」，而不是往目標移動。

#### ⚠️ 關鍵發現：Shaping × 10 低於 V(s) 估計噪音

加入 shaping 後，在 sim 環境（`ep_rew` 可達 +60）成功收斂，但在 Gazebo **仍然失敗**。

原因分析（2026-05 確認）：

Gazebo 無人機受物理阻尼，**實際位移 ≈ 0.05 m/step**（即使 1 m/s 指令），比理論值小。

| 指標 | Shaping × 10 | Shaping × 100 |
|------|-------------|--------------|
| 隨機 policy 的 episode return std | ≈ 9 | ≈ 90 |
| V(s) 估計誤差（典型值） | 20–50 | 20–50 |
| 信噪比 | **< 1 → 無法分辨** | **> 1 → 可學習** |

用 shaping × 10 時，V(s) 估計雜訊比 return 變異更大——PPO 無法區分「有效導航」與「原地徘徊」的 episode，policy gradient 等效為零。

同時，距離懲罰 `−dist × 2.0` 在 Phase 3+ 產生 −600 ~ −1200 的均一負 baseline，進一步壓制了 shaping 的相對影響力。

**修正**：
- `shaping × 100`：std(return) ≈ 90，信噪比 > 1
- `distance × 0.5`：Phase 5 baseline ≈ −300（vs 舊版 −1200），shaping 佔比更大

Shaping 符合 Ng et al. (1999) 的 potential-based 形式（`F = γΦ(s') − Φ(s)`，`Φ(s) = −dist`），不改變最優 policy，只加速收斂。

### 終止條件

| 條件 | 原因 |
|------|------|
| 30 步連續 dist < 0.3 m | 任務成功 |
| z < 0.1 m（step > 10）| 墜機 |
| `\|pos\|` > 6 m | 飛出範圍（Task E 用 10 m 導致卡牆，縮小） |
| 達到 500 步（sim）/ 300 步（Gazebo）| Timeout |

### 成功條件設計討論

初版：`dist < 0.3 AND vel < 0.3` 連續 30 步。
問題：noise 迫使 policy 持續送修正指令 → vel 不斷超過 0.3 → stable_count 不停重置。

修正版：**只看距離**，速度留在 per-step penalty 中。
```python
if dist < success_radius:   # 不再檢查 vel
    stable_count += 1
```

---

## 環境實作

### 快速模擬 (drone_env_sim.py)

**物理模型**：
```
alpha = DT / TAU = 0.1 / TAU
vel_t = (1 − alpha) × vel_{t-1} + alpha × action    # first-order lag
pos_t = pos_{t-1} + vel_t × DT
```

TAU 每個 episode 從 [0.05, 0.4] s 隨機取樣，讓 policy 對動力學不確定性具有魯棒性。

**擾動模型**：
```python
pos += N(0, noise_level × 0.05)  # 每步施加隨機位移
pos[2] = max(pos[2], 0.05)       # 防止穿地
```

**起始位置**：XY ∈ [−1, 1] m，Z ∈ [1, 3] m 隨機化。

---

### Pybullet 環境 (drone_env_pybullet.py)

使用 `gym-pybullet-drones` HoverAviary，搭配 VEL action type，以真實 CF2X 剛體物理模擬。

**校正參數**（對齊 Gazebo simple_drone）：

| 參數 | 預設值 | 修改後 | 原因 |
|------|--------|--------|------|
| SPEED_LIMIT | 0.25 m/s | **1.0 m/s** | 對齊 Gazebo cmd_vel 有效範圍 |
| 起始高度 | aviary 預設 | **z ≈ 2.0 m** | 對齊 Gazebo takeoff 後的懸停高度 |
| pybullet 終止條件 | aviary 內建 | **全部忽略** | aviary 的 EPISODE_LEN_SEC 和 hover-(0,0,1) 邏輯與本任務不符 |

**Target 隨機化**：訓練時在目標點 ±0.5 m 加擾動，提高泛化性。評估時關閉（`target_rand_range=0.0`）。

---

### Gazebo 真實環境 (drone_env.py)

ROS 2 interface：訂閱 `/simple_drone/gt_pose`，發布 `/simple_drone/cmd_vel`。
速度估計：有限差分 `vel = (pos_new − pos_old) × 10 Hz`。

---

## 訓練策略

### 本機快速模擬訓練

```bash
python3 train.py --sim --timesteps 1000000
```

| 超參數 | 值 |
|--------|----|
| learning_rate | 3e-4 |
| n_steps | 1024 |
| batch_size | 256 |
| n_epochs | 10 |
| γ | 0.99 |
| λ (GAE) | 0.95 |
| ent_coef | 0.01 |
| n_envs | 8 |

---

### Gazebo 課程學習 (train_gazebo.py) — 最終設計（2026-05）

#### 核心設計原則：V(s) 播種 + 高信噪比 Shaping

直接在 Gazebo 訓練的兩大根本挑戰：

1. **零 advantage 循環**：policy 無法到達 target → 所有 return 均一負值 → V(s) ≈ 常數 → Advantage ≈ 0 → policy gradient ≈ 0 → 永遠不學習。
2. **Shaping 信噪比不足**（2026-05 新發現）：shaping × 10 的 return std ≈ 9，低於 V(s) 估計誤差（20–50），PPO 無法分辨導航動作與徘徊動作的優劣。

**解法一：Phase 1 target = (0, 0, 2)，stable_steps = 1，僅 5k steps**

- Drone 起始點 = target，dist ≈ 0 < 0.3 m → 第 1 步立即觸發成功 → ep_rew ≈ +103，ep_len ≈ 1
- V(s) 被播種為 +100，為 Phase 2 建立強大 advantage 梯度
- 僅 5k steps（≈5000 個 1-step episode）：**不學習任何動作偏好，無 hover bias**

比較舊版（stable_steps=30）：需維持 30 步 → 學到主動修正漂移的行為 → 與導航衝突。

**解法二：shaping × 100，distance × 0.5（2026-05 新增）**

```
Phase 2 random policy return std：9（舊版 ×10）→ 90（新版 ×100）
Phase 5 baseline（hover at origin）：−1200（舊）→ −300（新 ×0.5 distance）
```

PPO 現在可以區分「有效導航」（return >> mean）與「徘徊」（return ≈ mean）。

**Phase 2 advantage 機制**（新版 reward + Phase 1 播種後）：

```
V(s₀) ≈ +100（Phase 1 播種）
Phase 2 hover-at-origin return ≈ −78（−0.5×0.5×300 − 3）
Advantage(zero-action) = −78 − 100 = −178   ← 強烈推離 zero-action

Phase 2 成功導航 return ≈ +163（shaping+zone+success）
Advantage(navigate)   = +163 − 100 = +63    ← 強烈正向強化
```

#### 五階段課程

```python
DEFAULT_CURRICULUM = [
    ((0.0, 0.0, 2.0),   5_000),   # Phase 1: 即時成功播種 V(s)≈+100
    ((0.5, 0.0, 2.0),  60_000),   # Phase 2: 0.5 m 導航，stable=5
    ((1.0, 0.0, 2.0),  80_000),   # Phase 3: 1.0 m 導航，stable=10
    ((1.5, 0.0, 2.0), 100_000),   # Phase 4: 1.5 m 導航，stable=20
    ((2.0, 0.0, 2.0), 200_000),   # Phase 5: 完整任務，stable=30
]
CURRICULUM_STABLE_STEPS = [1, 5, 10, 20, 30]
# 總計: 445k steps ≈ 3.5 hours in Gazebo
```

#### Phase 轉換策略：什麼都不做

```python
# 只更新 target、stable_steps_required 和 prev_dist
env.unwrapped._target               = np.array(target, dtype=np.float32)
env.unwrapped.stable_steps_required = stable_steps
env.unwrapped._prev_dist = float(np.linalg.norm(pos - env.unwrapped._target))
```

- **NO value function reset**：reset 讓所有 advantage = actual_return − 0 = 均一負值，policy 被懲罰所有動作，技能被抹掉。
- **NO LR 降低**：效果等同 reset，只是更慢。

#### PPO 超參數（From Scratch）

| 參數 | 值 | 說明 |
|-----|----|------|
| learning_rate | 3e-4 | 標準值 |
| n_steps | 2048 | Gazebo 較慢，大 batch 穩定 |
| batch_size | 64 | |
| n_epochs | 10 | |
| gamma | 0.99 | |
| gae_lambda | 0.95 | |
| ent_coef | **0.001** | 全負 reward 環境；0.005 仍可能使 std 爆炸 |
| device | cpu | |

#### 訓練指令

```bash
# Docker 裡面，先啟動 Gazebo，確認 drone 已 takeoff
python3 train_gazebo.py --ent-coef 0.001   # 從頭開始（建議）
python3 train_gazebo.py --ent-coef 0.001 --skip-phases 2  # 從 Phase 3 續訓
```

---

## 訓練紀錄

### Sim 訓練（1M steps）

```
python3 train.py --sim --timesteps 1000000
```

| 指標 | 值 |
|------|----|
| ep_len_mean | 49.6 |
| ep_rew_mean | +62.8 |
| explained_variance | 0.831 |
| std | 0.529 |
| 牆鐘時間 | ~165 s |

---

### Gazebo 課程訓練 — 完整失敗紀錄（按時間順序）

#### 失敗 1：ent_coef 過高（0.05）

```
ent_coef=0.05 → std 0.514 → 1.09 → 1.27 → 純隨機
ep_rew: −252 → −454 → −988（持續惡化）
```

全負 reward 環境下，entropy 項主導 loss，policy 輸出均一雜訊。

#### 失敗 2：Phase 1 = (0.5, 0, 2)，stable_steps=30 → zero-action local optimum

```
Phase 1 收斂：ep_rew=-8, ep_len=30
Phase 2 (target=1,0,2)：ep_rew=-710, ep_len=300
EV=0.995, policy_gradient≈0 → 卡死
```

#### 失敗 3：Value Function Reset at phase transition

```
Phase 1 V(s) 預測 +50 → reset 到 0 → advantage = return−0 = −300（全部動作）
→ 技能全部消失，value_loss 短暫 → 2.25×10⁹
```

#### 失敗 4：Phase 1 at (0,0,2)，stable_steps=30 → hover bias

```
Phase 1 ep_rew ≈ +57~+81（成功播種 V(s)）
Phase 2 iter 1: ep_rew=+39, ep_len=70（仍用 Phase 1 policy）
Phase 2 iter 8: ep_rew=-159, ep_len=183（轉移中）
Phase 2 iter 13-19: ep_rew=-273~-373, ep_len=253~299，clip_fraction→0（卡死）
```

根本原因：Phase 1 的 30-step 成功條件迫使 policy 學到「主動修正漂移→回 (0,0,2)」的行為。Phase 2 時這個 corrective policy 與導航衝突，drone 永遠往 (0,0,2) 修正。

#### 失敗 5：Phase 1 at (0.5,0,2)，stable_steps=1 → Gazebo 物理阻尼問題

理論上 random policy 應能 99% 機率在 300 步內到達 (0.5m zone)，但**實際 Gazebo 物理阻尼導致隨機 policy 的有效位移遠小於理論值**：

```
Phase 1 (target=0.5,0,2), stable_steps=1:
  iter 1-15: ep_len=300（全部 timeout，無一成功）
  ep_rew: -313 → -394（drone 在起始點徘徊，無法到達 zone）
Phase 2, 3, 4, 5: 全部 ep_rew 持續下降，drone 固定在 (0,0,2)
```

Phase 1 無成功 → V(s) 均一負值 → 零 advantage → Phase 2-5 全部失敗。

#### 失敗 6：Phase 1 at (0,0,2)，stable_steps=1，5k steps，但 shaping×10（最新一次失敗）

Phase 1 正確播種 V(s)≈+100（ep_len≈1，ep_rew≈+103），但 Phase 2+ 無效：

```
Phase 3 (target=1,0,2): ep_rew ≈ -618 (= hover-at-origin baseline: -1×2×300=-600)
Phase 4 (target=1.5,0,2): ep_rew -645 → -880（drone 固定在 (0,0,2)）
```

**根本原因（最終確認）**：shaping×10 在 Gazebo 物理阻尼下 return std ≈ 9，低於 V(s) 估計誤差（20–50）。PPO 無法從 return variance 中分辨導航行為的優劣。V(s) 在 Phase 2 的第 1 iteration 從 +100 迅速適應到 -300，advantage 窗口關閉，policy 停止更新。

#### 修正方案（2026-05）：shaping×100 + distance×0.5

- return std: 9 → 90（信噪比從 <1 提升到 >1）
- Phase 5 baseline: −1200 → −300
- 信號對噪音比顯著改善，PPO 可有效學習導航

---

### Gazebo 訓練結果匯總

| 嘗試 | Phase 1 設計 | Reward | 最終結果 |
|------|------------|--------|---------|
| ent_coef=0.05 | (0.5,0,2) stable=30 | −988 | std 爆炸 |
| ent_coef=0.001 | (0.5,0,2) stable=30 | −373 | hover bias 卡死 |
| V(s) reset | (0,0,2) stable=30 | −300 均一 | 技能消失 |
| stable_steps=[30,5...] | (0,0,2) stable=30 | −618 Phase3 | hover bias |
| stable_steps=[1,5...] | (0.5,0,2) stable=1 | −394 Phase1 | Gazebo 阻尼 |
| stable_steps=[1,5...]，shaping×10 | (0,0,2) stable=1 | −880 Phase4 | SNR < 1 |
| **stable_steps=[1,5...]，shaping×100** | **(0,0,2) stable=1** | **進行中** | **目標方案** |

---

## 測試結果

### Sim 環境（1M steps）

```
python3 test.py --sim --episodes 20
```

| 條件 | Success | Avg reward | Avg dist | Avg steps |
|------|---------|------------|----------|-----------|
| 無 noise | **20/20 (100%)** | +71.4 ± 14.4 | 0.039 m | 48 |

### Gazebo 真實環境

*(進行中：5-phase curriculum，shaping×100，Phase 1 at (0,0,2) stable=1，5k steps)*

---

## 與 Task E 對比

| 面向 | Task E（多路徑點） | Task A（懸停） |
|------|--------------------|----------------|
| 觀測維度 | 13-D | 10-D |
| 成功條件 | 進入 0.8 m 半徑 | 30 步 dist < 0.3 m |
| 獎勵設計 | Potential shaping | −dist + shaping |
| OOB 邊界 | 10 m | 6 m |
| 起始位置 | 固定 (0,0,2) | 隨機 ±1 m |
| Sim 收斂速度 | ~2M steps | **~100k steps** |
| Sim 成功率 | ~100% | **100%** |
| Gazebo 成功率 | **0/10 (0%)** | 待測 |

---

## 已知問題與改進方向

1. **Sim-to-Real gap**：模擬用 first-order lag（TAU≈0.2s），Gazebo 有完整剛體物理。Pybullet 環境（drone_env_pybullet.py）是中間橋樑，但 CF2X 與 simple_drone 仍有質量/慣量差異。

2. **Velocity threshold 移除**：只看距離成功，true hover 穩定性在 Gazebo 中可能更難達到。

3. **Domain randomisation**：模擬訓練對 TAU [0.05, 0.4] s 隨機化，有助縮小 sim-to-real 差距；Gazebo fine-tuning 也可加入 `--noise 0.5` 增強魯棒性。

4. **Phase 2 J-curve 現象**：Phase 1→2 轉換後 ep_rew 先下降（drone 學習新行為），預計 10–25k steps 後回升。這是正常的 curriculum learning 現象，不需要干預。

5. **V(s) 適應速度問題**：V(s) 在 phase 轉換後約 1–2 iteration（≈2000 steps）內重新適應至新 phase 的 return level，advantage 窗口極短。需搭配 shaping×100 確保 return variance 足夠大，讓 advantage 訊號在窗口期內可學習。

6. **Shaping 係數的 trade-off**：shaping×100 讓 return 對短距離位移也高度敏感。若 Gazebo 存在持續風場，drone 可能因「被風吹離目標」的 shaping 負值而學到反風補償行為（有益）；但若風向與目標方向衝突，可能讓訓練更複雜。
