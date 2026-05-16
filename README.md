# HW2 – NSYSU Drone RL: Task E Multi-Waypoint Cruising

**Algorithm:** PPO (Stable-Baselines3)  
**Task:** Drone must visit 5 ordered 3-D waypoints while minimizing time  
**Deadline:** 2026-05-30 23:59

---

## 1. Prerequisites

- Docker Desktop (Windows) or Docker 24.x (Linux/macOS)
- NVIDIA GPU + driver (optional but recommended)
- TurboVNC Viewer — download at https://github.com/TurboVNC/turbovnc/releases

---

## 2. Docker Setup

### Step 1 — Clone base simulation repo & build image

```bash
git clone https://github.com/NSYSU-ARL/Assignment2.git
cd Assignment2
docker build -t nsysu_drone_vnc:iron .
```

### Step 2 — Launch container

> **Note:** Do NOT use `--rm`. Without it, the container preserves installed packages
> and copied files across restarts — you only need to install dependencies once.

**First launch — Windows (PowerShell) with NVIDIA GPU:**
```powershell
docker run -it --gpus all -p 5901:5901 --privileged --name nsysu_drone_vnc nsysu_drone_vnc:iron
```

**First launch — Windows (PowerShell) without GPU:**
```powershell
docker run -it -p 5901:5901 --privileged --name nsysu_drone_vnc nsysu_drone_vnc:iron
```

**Subsequent launches (state preserved):**
```powershell
docker start -i nsysu_drone_vnc
```

**Linux/macOS (first launch):**
```bash
./run_docker.sh
```

### Step 3 — Connect VNC

Open TurboVNC Viewer → connect to `localhost:5901` → password: **nsysudrone**

### Step 4 — Launch Gazebo (inside VNC terminal)

```bash
launch_drone
```

---

## 3. Copy Assignment Code into Container

Open a second terminal (keep VNC open), then:

**Windows PowerShell:**
```powershell
docker cp part1 nsysu_drone_vnc:/root/
docker cp part2 nsysu_drone_vnc:/root/
```

**Linux/macOS:**
```bash
docker cp part1 nsysu_drone_vnc:/root/
docker cp part2 nsysu_drone_vnc:/root/
```

---

## 4. Install Python Dependencies (first time only)

Inside VNC terminal:

```bash
python3 -m pip install stable-baselines3 gymnasium tensorboard \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120
```

---

## 5. Part 1 – fly_straight (20 pts)

Inside VNC terminal (Gazebo must be running):

```bash
cd /root/part1

# Need takeoff before each run:
ros2 topic pub --once /simple_drone/takeoff std_msgs/msg/Empty "{}"

# Default target (5, 3, 2)
python3 fly_straight_part1.py 2>&1 | tee terminal_log.txt

# Alternative target 1: (-2, 4, 1.5)
python3 fly_straight_part1.py --target -2 4 1.5 2>&1 | tee -a terminal_log.txt

# Alternative target 2: (0, 0, 3)
python3 fly_straight_part1.py --target 0 0 3 2>&1 | tee -a terminal_log.txt

# Alternative target 3: (6, -3, 1)
python3 fly_straight_part1.py --target 6 -3 1 2>&1 | tee -a terminal_log.txt
```

**Kp study** (observe oscillation vs sluggishness):
```bash
python3 fly_straight_part1.py --target 5 3 2 --kp 0.1   # sluggish
python3 fly_straight_part1.py --target 5 3 2 --kp 0.5   # baseline
python3 fly_straight_part1.py --target 5 3 2 --kp 1.5   # oscillates
```

**max_speed study** (observe overshoot):
```bash
python3 fly_straight_part1.py --target 5 3 2 --max_speed 0.3  # slow/safe
python3 fly_straight_part1.py --target 5 3 2 --max_speed 1.0  # baseline
python3 fly_straight_part1.py --target 5 3 2 --max_speed 2.0  # overshoots
```

Save Gazebo screenshots to `part1/screenshot_*.png`.

---

## 6. Part 2 – RL Training (PPO)

```bash
cd /root/part2

# Warm-up: easy 4-waypoint square
python3 train.py --easy --timesteps 100000

# Full training: 5-waypoint medium set
python3 train.py --timesteps 300000
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
# Open browser at http://localhost:6006
```

Save the reward curve screenshot to `logs/training_curve.png`.

---

## 7. Part 2 – Evaluation

```bash
cd /root/part2

python3 test.py --model models/best/best_model --episodes 10
```

---

## 8. MDP Formulation

| Element | Definition |
|---------|-----------|
| **State S** | pos(3) + vel(3) + current waypoint(3) + distance(1) + progress(1) + XY direction(2) = **13-D** |
| **Action A** | velocity command (vx, vy, vz) ∈ [−1, 1]³ |
| **Reward R** | −0.01·dist − 0.005/step + 10·(WP reached) + 50·(all done) − 100·(crash) − 50·(OOB) |
| **γ** | 0.99 |

**Waypoints (WAYPOINTS_MEDIUM):**
1. (3, 0, 2)
2. (3, 3, 3)
3. (0, 3, 2.5)
4. (−2, 0, 2)
5. (0, −2, 1.5)

---

## 9. Submission Structure

```
HW2_<StudentID>_<Name>/
├── README.md
├── part1/
│   ├── fly_straight_part1.py
│   ├── screenshot_default.png
│   ├── screenshot_target1.png
│   ├── screenshot_target2.png
│   ├── screenshot_target3.png
│   └── terminal_log.txt
├── part2/
│   ├── drone_env.py
│   ├── train.py
│   ├── test.py
│   ├── models/
│   │   └── ppo_waypoint_final.zip
│   └── logs/
│       └── training_curve.png
└── HW2_<StudentID>_report.pdf
```
