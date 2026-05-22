#!/usr/bin/env python3
"""
test.py
-------
Evaluate a trained PPO model on Task E: Multi-Waypoint Cruising.

Usage (inside Docker container, after launching Gazebo with `launch_drone`):
    python3 test.py                                    # uses best model
    python3 test.py --model models/ppo_waypoint_final  # specify model path
    python3 test.py --episodes 20                      # more eval episodes
"""

import argparse
import sys
import types

import numpy as np

# Compatibility shim: models saved with NumPy 2.x reference numpy._core,
# which does not exist in NumPy 1.x (Docker container). Alias it before
# stable-baselines3 tries to unpickle the model.
if not hasattr(np, "_core"):
    _core_mod = types.ModuleType("numpy._core")
    for _attr in dir(np.core):
        try:
            _submod = getattr(np.core, _attr)
            setattr(_core_mod, _attr, _submod)
            sys.modules[f"numpy._core.{_attr}"] = _submod
        except Exception:
            pass
    np._core = _core_mod
    sys.modules["numpy._core"] = _core_mod

import rclpy
from stable_baselines3 import PPO

from drone_env import DroneROSInterface, DroneWaypointEnv, WAYPOINTS_MEDIUM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="models/best/best_model",
        help="Path to the saved PPO model (without .zip extension)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    args = parser.parse_args()

    rclpy.init()
    ros = DroneROSInterface()
    env = DroneWaypointEnv(ros_interface=ros, waypoints=WAYPOINTS_MEDIUM)

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    total_waypoints = len(WAYPOINTS_MEDIUM)
    rewards    = []
    wp_counts  = []
    step_counts = []
    successes  = 0

    print(f"\nRunning {args.episodes} evaluation episodes …\n")
    print(f"{'Ep':>4}  {'Reward':>9}  {'WP Done':>8}  {'Steps':>7}  Status")
    print("─" * 48)

    for ep in range(args.episodes):
        obs, _    = env.reset()
        total_rew = 0.0
        steps     = 0
        done      = False
        info      = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total_rew += rew
            steps     += 1
            done       = terminated or truncated

        wp_done = info.get("waypoints_done", 0)
        success = wp_done == total_waypoints

        if success:
            successes += 1
        rewards.append(total_rew)
        wp_counts.append(wp_done)
        step_counts.append(steps)

        status = "SUCCESS" if success else f"{wp_done}/{total_waypoints} WP"
        print(f"{ep+1:>4}  {total_rew:>9.1f}  {wp_done:>4}/{total_waypoints:<3}  "
              f"{steps:>7}  {status}")

    print("\n─── Summary ─────────────────────────────────────")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Success rate:   {successes}/{args.episodes} "
          f"({100 * successes / args.episodes:.0f}%)")
    print(f"  Avg reward:     {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg WP done:    {np.mean(wp_counts):.1f} / {total_waypoints}")
    print(f"  Avg steps:      {np.mean(step_counts):.0f}")

    env.close()
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
