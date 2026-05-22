#!/usr/bin/env python3
"""
train.py
--------
PPO training for Task E: Multi-Waypoint Cruising.

── Simulation mode (fast, no Docker/Gazebo needed) ──────────────────────────
    python3 train.py --sim                          # 8 parallel envs, 1M steps
    python3 train.py --sim --easy                   # easy waypoints warm-up
    python3 train.py --sim --timesteps 2000000      # longer run
    python3 train.py --sim --n-envs 16              # more parallel envs

── ROS 2 / Gazebo mode (inside Docker, after launch_drone) ──────────────────
    python3 train.py                        # 300k steps, medium waypoints
    python3 train.py --easy                 # 4-waypoint square, good warm-up
    python3 train.py --timesteps 500000     # longer run for better convergence

Monitor with TensorBoard:
    tensorboard --logdir logs/
Then save the reward curve screenshot to logs/training_curve.png.
"""

import argparse
import os
import sys
import types

# ── NumPy 1.x / 2.x compatibility shim ──────────────────────────────────────
# Models saved with NumPy 2.x reference numpy._core; NumPy 1.x (Docker) only
# has numpy.core.  Add aliases before stable-baselines3 tries to unpickle.
import numpy as _np
if not hasattr(_np, "_core"):
    import numpy.core.numeric     as _nc_numeric
    import numpy.core.multiarray  as _nc_multiarray
    import numpy.core.fromnumeric as _nc_fromnumeric
    _cm = types.ModuleType("numpy._core")
    _cm.numeric     = _nc_numeric
    _cm.multiarray  = _nc_multiarray
    _cm.fromnumeric = _nc_fromnumeric
    _np._core = _cm
    sys.modules["numpy._core"]             = _cm
    sys.modules["numpy._core.numeric"]     = _nc_numeric
    sys.modules["numpy._core.multiarray"]  = _nc_multiarray
    sys.modules["numpy._core.fromnumeric"] = _nc_fromnumeric
# ─────────────────────────────────────────────────────────────────────────────

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total environment steps (default: 1M for sim, 300k for ROS)"
    )
    parser.add_argument(
        "--easy", action="store_true",
        help="Use 4-waypoint square (WAYPOINTS_EASY) instead of WAYPOINTS_MEDIUM"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Train in fast simulation mode — no ROS 2 / Gazebo needed"
    )
    parser.add_argument(
        "--n-envs", type=int, default=8,
        help="Parallel environments for sim training (ignored in ROS mode)"
    )
    parser.add_argument(
        "--load", default=None,
        help="Path to a pre-trained model to fine-tune (without .zip extension)"
    )
    args = parser.parse_args()

    if args.timesteps is None:
        args.timesteps = 1_000_000 if args.sim else 300_000

    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    if args.sim:
        _train_sim(args)
    else:
        _train_ros(args)


# ── Simulation training (fast, no ROS 2) ────────────────────────────────────

def _train_sim(args) -> None:
    from stable_baselines3.common.env_util import make_vec_env
    from drone_env_sim import (
        DroneWaypointSimEnv, WAYPOINTS_EASY, WAYPOINTS_MEDIUM
    )

    waypoints = WAYPOINTS_EASY if args.easy else WAYPOINTS_MEDIUM
    wp_label  = "EASY (4 WP)" if args.easy else "MEDIUM (5 WP)"

    env_kwargs = {"waypoints": waypoints}

    # Vectorised training envs (each wrapped with Monitor automatically)
    train_env = make_vec_env(
        DroneWaypointSimEnv,
        n_envs=args.n_envs,
        env_kwargs=env_kwargs,
        monitor_dir="logs/",
    )
    # Single eval env
    eval_env = Monitor(DroneWaypointSimEnv(**env_kwargs), filename="logs/eval_monitor")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,           # steps per env per update; total = 1024 * n_envs
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        device="cpu",
        verbose=1,
        tensorboard_log="./logs/",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_sim",
    )
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=50_000,
        n_eval_episodes=10,
        best_model_save_path="./models/best/",
        deterministic=True,
        verbose=1,
    )

    print(f"[SIM] Starting PPO — {args.timesteps:,} timesteps  "
          f"({args.n_envs} parallel envs)  waypoints: {wp_label}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model …")
    finally:
        model.save("models/ppo_waypoint_final")
        print("Saved: models/ppo_waypoint_final.zip")
        train_env.close()
        eval_env.close()
        _plot_training_curve()


# ── ROS 2 / Gazebo training ──────────────────────────────────────────────────

def _train_ros(args) -> None:
    import rclpy
    from drone_env import (
        DroneROSInterface, DroneWaypointEnv, WAYPOINTS_EASY, WAYPOINTS_MEDIUM
    )

    waypoints = WAYPOINTS_EASY if args.easy else WAYPOINTS_MEDIUM
    wp_label  = "EASY (4 WP)" if args.easy else "MEDIUM (5 WP)"

    rclpy.init()
    ros = DroneROSInterface()

    env = Monitor(DroneWaypointEnv(ros_interface=ros, waypoints=waypoints),
                  filename="logs/monitor")

    if args.load:
        print(f"Fine-tuning from: {args.load}")
        model = PPO.load(args.load, env=env, device="cpu",
                         tensorboard_log="./logs/")
        model.ent_coef = 0.005   # less exploration when fine-tuning
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,   # higher: force exploration in Gazebo dynamics
            device="cpu",
            verbose=1,
            tensorboard_log="./logs/",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/",
        name_prefix="ppo_waypoint",
    )
    eval_cb = EvalCallback(
        env,
        eval_freq=10_000,
        n_eval_episodes=5,
        best_model_save_path="./models/best/",
        deterministic=True,
        verbose=1,
    )

    print(f"[ROS] Starting PPO — {args.timesteps:,} timesteps  waypoints: {wp_label}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model …")
    finally:
        model.save("models/ppo_waypoint_final")
        print("Saved: models/ppo_waypoint_final.zip")
        env.close()
        ros.destroy_node()
        rclpy.shutdown()
        _plot_training_curve()


# ── Training curve plot ──────────────────────────────────────────────────────

def _plot_training_curve() -> None:
    import glob
    import csv
    import matplotlib.pyplot as plt

    # Collect all monitor CSVs (single-env Monitor writes *.monitor.csv)
    csv_files = sorted(glob.glob("logs/*.monitor.csv"))
    if not csv_files:
        return

    timesteps, rewards = [], []
    with open(csv_files[0], newline="") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        t = 0
        for row in reader:
            t += int(float(row["l"]))
            timesteps.append(t)
            rewards.append(float(row["r"]))

    if not rewards:
        return

    window   = min(20, len(rewards))
    smoothed = [sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
                for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards,  alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(timesteps, smoothed, color="steelblue", linewidth=2, label="Smoothed (w=20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Curve — Task E Multi-Waypoint Cruising")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/training_curve.png", dpi=150)
    print("Training curve saved to logs/training_curve.png")


if __name__ == "__main__":
    main()
