#!/usr/bin/env python3
"""
test.py
-------
Evaluate a trained PPO model on Task A: Precision Hovering.

Usage (fast sim — no ROS 2 / Gazebo needed):
    python3 test.py --sim                              # uses best model
    python3 test.py --sim --model models/ppo_hover_final
    python3 test.py --sim --episodes 20
    python3 test.py --sim --noise 1.0                  # test disturbance rejection

Usage (pybullet — realistic physics, no ROS needed):
    python3 test.py --pybullet
    python3 test.py --pybullet --model models/ppo_hover_pb_final

Usage (Gazebo, inside Docker after launch_drone):
    python3 test.py
    python3 test.py --model models/ppo_hover_final
    python3 test.py --model models/ppo_hover_pb_final   # pybullet-trained policy (no scale needed)
"""

import argparse
import sys
import types

import numpy as np

# NumPy 1.x / 2.x compatibility shim (see train.py for explanation)
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

from stable_baselines3 import PPO


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
    parser.add_argument(
        "--sim", action="store_true",
        help="Evaluate in fast first-order-lag simulation — no ROS needed"
    )
    parser.add_argument(
        "--pybullet", action="store_true",
        help="Evaluate with pybullet Crazyflie physics — no ROS needed"
    )
    parser.add_argument(
        "--action-scale", type=float, default=1.0,
        help="Multiply policy actions by this factor before sending to Gazebo "
             "(use ~4.0 when deploying a pybullet-trained policy to Gazebo)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Drift noise level for sim evaluation (default: 0.0 — no disturbance)"
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=[2.0, 0.0, 2.0],
        metavar=("X", "Y", "Z"),
        help="Hover target position (default: 2 0 2)"
    )
    args = parser.parse_args()

    if args.sim:
        _eval_sim(args)
    elif args.pybullet:
        _eval_pybullet(args)
    else:
        _eval_ros(args)


# ── Simulation evaluation ────────────────────────────────────────────────────

def _eval_sim(args) -> None:
    from drone_env_sim import DroneHoverSimEnv

    target = tuple(args.target)
    env = DroneHoverSimEnv(target=target, noise_level=args.noise)

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    rewards       = []
    final_dists   = []
    step_counts   = []
    stable_counts = []
    successes     = 0

    noise_tag = f"  noise={args.noise}" if args.noise > 0 else ""
    print(f"\nEvaluating {args.episodes} episodes  target={target}{noise_tag}\n")
    print(f"{'Ep':>4}  {'Reward':>9}  {'FinalDist':>10}  {'Steps':>7}  {'StablePk':>9}  Status")
    print("─" * 56)

    for ep in range(args.episodes):
        obs, _    = env.reset()
        total_rew = 0.0
        steps     = 0
        done      = False
        info      = {}
        peak_stable = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total_rew   += rew
            steps       += 1
            peak_stable  = max(peak_stable, info.get("stable_count", 0))
            done         = terminated or truncated

        success = info.get("success", False)
        if success:
            successes += 1

        dist = info.get("distance", float("nan"))
        rewards.append(total_rew)
        final_dists.append(dist)
        step_counts.append(steps)
        stable_counts.append(peak_stable)

        status = "SUCCESS" if success else "timeout"
        print(f"{ep+1:>4}  {total_rew:>9.1f}  {dist:>10.3f}  {steps:>7}  "
              f"{peak_stable:>9}  {status}")

    env.close()

    print("\n─── Summary ─────────────────────────────────────")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Success rate:    {successes}/{args.episodes} "
          f"({100 * successes / args.episodes:.0f}%)")
    print(f"  Avg reward:      {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg final dist:  {np.mean(final_dists):.3f} m")
    print(f"  Avg steps:       {np.mean(step_counts):.0f}")
    print(f"  Avg peak stable: {np.mean(stable_counts):.1f} steps")


# ── Pybullet evaluation ───────────────────────────────────────────────────────

def _eval_pybullet(args) -> None:
    from drone_env_pybullet import DroneHoverPybulletEnv

    target = tuple(args.target)
    env = DroneHoverPybulletEnv(
        target=target,
        noise_level=args.noise,
        target_rand_range=0.0,   # fixed target for evaluation
    )

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    rewards       = []
    final_dists   = []
    step_counts   = []
    stable_counts = []
    successes     = 0

    noise_tag = f"  noise={args.noise}" if args.noise > 0 else ""
    print(f"\n[PYBULLET] Evaluating {args.episodes} episodes  "
          f"target={target}{noise_tag}\n")
    print(f"{'Ep':>4}  {'Reward':>9}  {'FinalDist':>10}  {'Steps':>7}  "
          f"{'StablePk':>9}  Status")
    print("─" * 56)

    for ep in range(args.episodes):
        obs, _    = env.reset()
        total_rew = 0.0
        steps     = 0
        done      = False
        info      = {}
        peak_stable = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total_rew   += rew
            steps       += 1
            peak_stable  = max(peak_stable, info.get("stable_count", 0))
            done         = terminated or truncated

        success = info.get("success", False)
        if success:
            successes += 1

        dist = info.get("distance", float("nan"))
        rewards.append(total_rew)
        final_dists.append(dist)
        step_counts.append(steps)
        stable_counts.append(peak_stable)

        status = "SUCCESS" if success else "timeout"
        print(f"{ep+1:>4}  {total_rew:>9.1f}  {dist:>10.3f}  {steps:>7}  "
              f"{peak_stable:>9}  {status}")

    env.close()

    print("\n─── Summary ─────────────────────────────────────")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Success rate:    {successes}/{args.episodes} "
          f"({100 * successes / args.episodes:.0f}%)")
    print(f"  Avg reward:      {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg final dist:  {np.mean(final_dists):.3f} m")
    print(f"  Avg steps:       {np.mean(step_counts):.0f}")
    print(f"  Avg peak stable: {np.mean(stable_counts):.1f} steps")


# ── ROS 2 / Gazebo evaluation ────────────────────────────────────────────────

def _eval_ros(args) -> None:
    import rclpy
    from drone_env import DroneROSInterface, DroneHoverEnv

    target = tuple(args.target)

    rclpy.init()
    ros = DroneROSInterface()
    env = DroneHoverEnv(ros_interface=ros, target=target)

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    rewards       = []
    final_dists   = []
    step_counts   = []
    stable_counts = []
    successes     = 0

    print(f"\nEvaluating {args.episodes} episodes  target={target}\n")
    print(f"{'Ep':>4}  {'Reward':>9}  {'FinalDist':>10}  {'Steps':>7}  {'StablePk':>9}  Status")
    print("─" * 56)

    for ep in range(args.episodes):
        obs, _    = env.reset()
        total_rew = 0.0
        steps     = 0
        done      = False
        info      = {}
        peak_stable = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action * args.action_scale, -1.0, 1.0)
            obs, rew, terminated, truncated, info = env.step(action)
            total_rew   += rew
            steps       += 1
            peak_stable  = max(peak_stable, info.get("stable_count", 0))
            done         = terminated or truncated

        success = info.get("success", False)
        if success:
            successes += 1

        dist = info.get("distance", float("nan"))
        rewards.append(total_rew)
        final_dists.append(dist)
        step_counts.append(steps)
        stable_counts.append(peak_stable)

        status = "SUCCESS" if success else "timeout"
        print(f"{ep+1:>4}  {total_rew:>9.1f}  {dist:>10.3f}  {steps:>7}  "
              f"{peak_stable:>9}  {status}")

    env.close()
    ros.destroy_node()
    rclpy.shutdown()

    print("\n─── Summary ─────────────────────────────────────")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Success rate:    {successes}/{args.episodes} "
          f"({100 * successes / args.episodes:.0f}%)")
    print(f"  Avg reward:      {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Avg final dist:  {np.mean(final_dists):.3f} m")
    print(f"  Avg steps:       {np.mean(step_counts):.0f}")
    print(f"  Avg peak stable: {np.mean(stable_counts):.1f} steps")


if __name__ == "__main__":
    main()
