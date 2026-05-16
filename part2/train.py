#!/usr/bin/env python3
"""
train.py
--------
PPO training for Task E: Multi-Waypoint Cruising.

Usage (inside Docker container, after launching Gazebo with `launch_drone`):
    python3 train.py                        # 300k steps, medium waypoints
    python3 train.py --easy                 # 4-waypoint square, good warm-up
    python3 train.py --timesteps 500000     # longer run for better convergence

Monitor with TensorBoard:
    tensorboard --logdir logs/
Then save the reward curve screenshot to logs/training_curve.png.
"""

import argparse
import os

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneROSInterface, DroneWaypointEnv, WAYPOINTS_EASY, WAYPOINTS_MEDIUM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Total environment steps to train for (default: 300000)"
    )
    parser.add_argument(
        "--easy", action="store_true",
        help="Use 4-waypoint square (WAYPOINTS_EASY) instead of WAYPOINTS_MEDIUM"
    )
    args = parser.parse_args()

    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    rclpy.init()
    ros = DroneROSInterface()

    waypoints = WAYPOINTS_EASY if args.easy else WAYPOINTS_MEDIUM
    env = Monitor(DroneWaypointEnv(ros_interface=ros, waypoints=waypoints))

    model = PPO(
        "MlpPolicy",
        env,
        # ── hyperparameters ──────────────────────────────────────────
        learning_rate=3e-4,
        n_steps=2048,       # steps collected per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,      # entropy bonus encourages exploration
        # ── logging ─────────────────────────────────────────────────
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

    print(f"Starting PPO training — {args.timesteps:,} timesteps")
    print(f"Waypoint set: {'EASY (4 WP)' if args.easy else 'MEDIUM (5 WP)'}")

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


if __name__ == "__main__":
    main()
