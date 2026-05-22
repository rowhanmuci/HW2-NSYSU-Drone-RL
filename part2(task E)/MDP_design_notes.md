# MDP Design Notes — Task E Multi-Waypoint Cruising

## Final MDP Formulation

| Element | Definition |
|---------|-----------|
| **State S** | 13-D: pos(3) + vel(3) + current_wp(3) + dist(1) + progress(1) + xy_dir(2) |
| **Action A** | velocity command (vx, vy, vz) ∈ [−1, 1]³ m/s |
| **γ** | 0.99 |
| **Episode max** | 300 steps (real) / 500 steps (sim) |
| **Success radius** | 0.8 m |

### Reward Function

```
r = (prev_dist − dist) × 2.0      # potential-based shaping (dense)
  − 0.01                           # time penalty per step
  + 20.0   if dist < 0.8 m        # waypoint reached
  + 100.0  if all waypoints done   # full-course bonus  → terminated
  − 100.0  if z < 0.1 m           # crash (after step 20 grace period)
  − 50.0   if |xy| > 10 m or z > 10 m  # out of bounds (after step 20)
```

### Waypoints (WAYPOINTS_MEDIUM — used for final training)

| # | x | y | z |
|---|---|---|---|
| 1 | 3.0 | 0.0 | 2.0 |
| 2 | 3.0 | 3.0 | 3.0 |
| 3 | 0.0 | 3.0 | 2.5 |
| 4 | −2.0 | 0.0 | 2.0 |
| 5 | 0.0 | −2.0 | 1.5 |

### Waypoints (WAYPOINTS_EASY — warm-up / fast debugging)

| # | x | y | z |
|---|---|---|---|
| 1 | 2.0 | 0.0 | 2.0 |
| 2 | 4.0 | 2.0 | 2.0 |
| 3 | 2.0 | 4.0 | 2.0 |
| 4 | 0.0 | 2.0 | 2.0 |

---

## Design Iterations & Lessons Learned

### Attempt 1 — Distance penalty (failed)
```python
reward = -0.01 * dist - 0.005   # per step
```
**Problem:** ep_len=1, reward=−51 — drone crashed at step 1 before it could take off.  
**Fix:** Added 10-step grace period before crash/OOB checks.

---

### Attempt 2 — Grace period added, but reward still flat (failed)
```python
reward = -0.01 * dist - 0.005
if dist < radius: reward += 10; if all done: reward += 50
```
**Problem:** `max_steps=1000` → only ~100 episodes per 100k steps; value function never converged (ep_rew=−130 flat).  
**Fix:** Reduced `max_steps` to 300; increased waypoint reward to +20, all-done to +100.

---

### Attempt 3 — Potential-based shaping (working)
```python
reward = (prev_dist - dist) * 2.0 - 0.01
```
**Why it works:**
- Dense reward at every step proportional to progress → much better gradient signal
- Drone gets immediate positive reward just for moving toward waypoint
- Avoids sparse reward problem of distance-based formulation

**Problem:** reward=−3 flat (ep_len=300) → drone appeared stationary.  
**Root cause:** `time.sleep()` in `reset_and_takeoff()` blocked ROS callbacks → `current_pose` was stale after reset.  
**Fix:** Replaced `time.sleep()` with `rclpy.spin_once()` loops.

---

### Attempt 4 — Spin-loop reset (partially working)
**Problem:** Reward jumped to −50 at ep_len=11 — OOB penalty at first step after grace period.  
**Root cause:** `/simple_drone/reset` does NOT teleport drone in Gazebo; it only resets state flags. After OOB crash, drone stays at OOB position. After takeoff, drone is still outside bounds → OOB at step 11 → every subsequent episode starts from OOB → cascade failure.  
**Fix:** Added P-controller after takeoff to fly drone back to (0, 0, 2) before returning from `reset_and_takeoff()`. Increased grace period from 10 to 20 steps.

---

### Sim-to-Real Gap

Trained PPO in pure-Python simulation (`drone_env_sim.py`) then transferred to Gazebo.

**Sim training results:**
- EASY (4 WP): reward=194, ep_len=61 steps — converged in 500k steps, ~75 sec
- MEDIUM (5 WP): reward=221, ep_len=101 steps — converged in 2M steps, ~315 sec

**Gazebo direct eval result:** 0/10 success, reward=−3, ep_len=300  
**Root cause:** Dynamics gap — sim uses first-order lag (TAU=0.2 s), Gazebo has full rigid-body physics. Sim policy outputs small actions that don't produce expected movement.

---

### Gazebo Training Attempts (all failed to converge)

#### Attempt G1 — Fine-tune from sim model (ent_coef=0.005)
- 100k steps, reward stuck at −3, ep_len=300
- **Root cause:** Policy already stuck in "stay still = −3" local optimum. With low entropy, no exploration to escape.

#### Attempt G2 — Train from scratch, ent_coef=0.05, out_of_bounds=10
- Initial improvement: −21.9 → −14.1 → −10.9 (steps 0–6k)
- Then cascaded back to −3 after drone flew OOB and got stuck at Gazebo room wall
- **Root cause:** Gazebo room wall (~6–8 m) is inside our 10 m OOB boundary. Drone hits wall, physics prevents movement, shaping = 0, reward = −3 exactly. Policy then maximises entropy (std kept growing: 2.0 → 8.7).

#### Attempt G3 — out_of_bounds=5.0 (too small)
- Reward immediately stuck at −3
- **Root cause:** Policy learned "stay still to avoid −50 OOB penalty". 5 m boundary too tight — any movement risks termination.

#### Attempt G4 — out_of_bounds=8.0, success_radius=1.5
- std grew: 7.2 → 8.7 over 200k steps; reward stayed −3 ± 0.1
- **Root cause:** Even with larger radius, drone never accidentally reaches waypoint in 300 steps. No positive gradient signal → full entropy maximisation.

### Summary of Why Gazebo RL Failed

| Factor | Effect |
|--------|--------|
| Physical room walls inside OOB boundary | Drone gets stuck; shaping=0; reward=−3 |
| Slow training (~10–38 fps) | Few episodes, rare waypoint encounters by chance |
| Sparse positive reward (never reaches WP) | No gradient to improve; policy collapses to entropy max |
| Dynamics gap (sim TAU≠Gazebo) | Sim-trained actions ineffective in Gazebo |

### Remaining Fix Ideas (not yet tried)
- Domain randomisation over TAU in sim (0.1–0.5 s)
- VecNormalize observation normalisation
- Curriculum: single waypoint at 0.5 m, grow distance gradually
- Reduce success_radius to check Gazebo room exact dimensions first

---

## PPO Hyperparameters

### Simulation mode
| Param | Value | Rationale |
|-------|-------|-----------|
| learning_rate | 3e-4 | Standard PPO default |
| n_steps | 1024 | Per env; total = 1024 × 8 envs = 8192 |
| batch_size | 256 | Larger for stable updates with many envs |
| n_epochs | 10 | Standard |
| gamma | 0.99 | Long-horizon planning |
| gae_lambda | 0.95 | Bias-variance tradeoff |
| ent_coef | 0.005 | Low: sim is deterministic, less exploration needed |
| n_envs | 8 | Parallel envs for speed |

### Gazebo mode (fine-tune)
| Param | Value | Rationale |
|-------|-------|-----------|
| n_steps | 2048 | Only 1 env, larger buffer per update |
| batch_size | 64 | Single env, smaller OK |
| ent_coef | 0.005 | Preserve learned policy structure |

---

## Observation Space (13-D)

| Index | Feature | Why included |
|-------|---------|--------------|
| 0–2 | Position (x, y, z) | Agent needs to know where it is |
| 3–5 | Velocity (vx, vy, vz) | Needed for smooth control (prevents oscillation) |
| 6–8 | Current waypoint (wx, wy, wz) | Target location |
| 9 | Distance to waypoint | Scalar progress signal |
| 10 | Progress ratio (done/total) | Contextualises which waypoint is next |
| 11–12 | XY unit direction to waypoint | Shortcut hint for policy; avoids learning atan2 |
