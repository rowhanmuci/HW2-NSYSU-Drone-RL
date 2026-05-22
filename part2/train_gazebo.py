#!/usr/bin/env python3
"""
train_gazebo.py
---------------
Curriculum-based PPO training directly in Gazebo (ROS 2).

Design principle — navigation-first, stability-second:

  Curriculum is over the SUCCESS CONDITION (stable_steps), not the target
  distance.  A random policy starting at (0,0,2) visits the 0.3 m zone
  around (0.5,0,2) with ~99% probability per 300-step episode (Brownian
  motion; ~7 expected zone steps per episode).  With stable_steps=1, the
  first phase sees nearly 100% episode success → massive positive V(s)
  seeding → the policy learns to navigate toward the target.
  Subsequent phases tighten the hover requirement rather than moving the
  target far away, so the policy always starts with transferable skills.

  Contrast with the FAILED "hover-at-(0,0,2)" Phase 1 approach:
  That seeded positive V(s) but also trained an active corrective policy
  that fights Phase-2 navigation (drone drifts toward 0.5 m, Phase-1
  behavior pulls it back to 0 m).

Curriculum phases (default):
  Phase 1  target=(0.0,0,2)   5 k steps  stable_steps=1   — instant success, seeds V(s)
  Phase 2  target=(0.5,0,2)  60 k steps  stable_steps=5   — navigate 0.5 m
  Phase 3  target=(1.0,0,2)  80 k steps  stable_steps=10  — navigate 1 m
  Phase 4  target=(1.5,0,2) 100 k steps  stable_steps=20  — navigate 1.5 m
  Phase 5  target=(2.0,0,2) 200 k steps  stable_steps=30  — full task
  Total: 445 k steps ≈ 3.5 hours in Gazebo

Usage (inside Docker, after launch_drone):
    python3 train_gazebo.py               # from scratch — recommended
    python3 train_gazebo.py --skip-phases 2   # resume at phase 3
"""

import argparse
import os
import sys
import types

import numpy as np

# ── NumPy 1.x / 2.x compatibility shim ──────────────────────────────────────
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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ── Curriculum definition ────────────────────────────────────────────────────

# Each entry: (target_xyz, timesteps)
# Phase 1: target=(0,0,2) = drone's start position, stable_steps=1.
# Drone starts INSIDE the zone → success on step 1 → ep_len≈1 → ep_rew≈+103.
# Only 5k steps = ~5000 episodes, each 1 step long.
# Policy NEVER has time to learn drift correction (episodes too short),
# so NO hover bias is created, only a strong V(s)≈+100 prior.
DEFAULT_CURRICULUM = [
    ((0.0, 0.0, 2.0),   5_000),
    ((0.5, 0.0, 2.0),  60_000),
    ((1.0, 0.0, 2.0),  80_000),
    ((1.5, 0.0, 2.0), 100_000),
    ((2.0, 0.0, 2.0), 200_000),
]

# Stable-hover requirement tightens across phases.
# Phase 1 stable_steps=1 → success on step 1 (drone starts at target).
# V(s)≈+100 seeded → Phase 2 advantage(zero-action) ≈ −403 → policy
# strongly pushed toward navigation.
CURRICULUM_STABLE_STEPS = [1, 5, 10, 20, 30]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum PPO training for Task A in Gazebo"
    )
    parser.add_argument(
        "--load", default=None,
        help="Resume from a saved checkpoint (skip early phases with --skip-phases)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training steps across all phases, distributed proportionally. "
             "Default: 470 k (30+60+80+100+200)."
    )
    parser.add_argument(
        "--skip-phases", type=int, default=0,
        metavar="N",
        help="Skip the first N phases (0 = start from phase 1)."
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.005,
        help="Entropy coefficient (default: 0.005). "
             "Keep ≤0.005 to prevent std explosion in Gazebo."
    )
    args = parser.parse_args()

    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    import rclpy
    from drone_env import DroneROSInterface, DroneHoverEnv

    rclpy.init()
    ros = DroneROSInterface()

    # Scale curriculum timesteps if --timesteps specified
    curriculum = DEFAULT_CURRICULUM[args.skip_phases:]
    if args.timesteps is not None:
        default_total = sum(s for _, s in curriculum)
        scale = args.timesteps / default_total
        curriculum = [(t, max(1000, int(s * scale))) for t, s in curriculum]

    if not curriculum:
        print("All phases skipped — nothing to train.")
        rclpy.shutdown()
        return

    first_target = curriculum[0][0]
    env = Monitor(
        DroneHoverEnv(ros_interface=ros, target=first_target),
        filename="logs/gazebo_curriculum_monitor",
    )

    # ── Model initialisation ─────────────────────────────────────────────────
    if args.load:
        print(f"\n[INFO] Resuming from: {args.load}")
        model = PPO.load(args.load, env=env, device="cpu",
                         tensorboard_log="./logs/")
        model.ent_coef = args.ent_coef
        print(f"[INFO] ent_coef → {args.ent_coef}")
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
            ent_coef=args.ent_coef,
            device="cpu",
            verbose=1,
            tensorboard_log="./logs/",
        )

    # ── Curriculum loop ──────────────────────────────────────────────────────
    total_phases = len(DEFAULT_CURRICULUM)
    for rel_idx, (target, steps) in enumerate(curriculum):
        phase_num = args.skip_phases + rel_idx + 1
        print(f"\n{'='*55}")
        print(f"  Phase {phase_num}/{total_phases}  —  target={target}  —  {steps:,} steps")
        print(f"{'='*55}")

        # Update target, shaping baseline, and success threshold for this phase
        stable_steps = CURRICULUM_STABLE_STEPS[phase_num - 1]
        env.unwrapped._target               = np.array(target, dtype=np.float32)
        env.unwrapped.stable_steps_required = stable_steps
        pos = ros.current_pose
        env.unwrapped._prev_dist = float(np.linalg.norm(pos - env.unwrapped._target))
        print(f"  stable_steps_required → {stable_steps}")

        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path="./models/",
            name_prefix=f"ppo_gz_phase{phase_num}",
        )

        try:
            model.learn(
                total_timesteps=steps,
                callback=checkpoint_cb,
                reset_num_timesteps=(phase_num == 1),
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving current model …")
            model.save(f"models/ppo_gz_phase{phase_num}_interrupted")
            break

        save_path = f"models/ppo_gz_phase{phase_num}"
        model.save(save_path)
        print(f"  Saved: {save_path}.zip")

        # Phase transition: do nothing special.
        # V(s) keeps its existing predictions; critic adapts naturally
        # to the new target over the first few thousand steps.

    # Final model always saved so test.py can find it
    model.save("models/ppo_gz_final")
    print("\nFinal model saved: models/ppo_gz_final.zip")

    env.close()
    ros.destroy_node()
    rclpy.shutdown()
    _plot_training_curve()


# ── Training curve plot ──────────────────────────────────────────────────────

def _plot_training_curve() -> None:
    import glob
    import csv
    import matplotlib.pyplot as plt

    csv_files = sorted(glob.glob("logs/gazebo_curriculum_monitor*.csv"))
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

    window   = min(10, len(rewards))
    smoothed = [sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
                for i in range(len(rewards))]

    phase_steps = [t for _, t in DEFAULT_CURRICULUM]
    total_default = sum(phase_steps)
    total_actual  = max(timesteps) if timesteps else total_default
    boundaries = [
        int(sum(phase_steps[:i+1]) / total_default * total_actual)
        for i in range(len(phase_steps) - 1)
    ]

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, rewards,  alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(timesteps, smoothed, color="steelblue", linewidth=2, label="Smoothed (w=10)")
    for b in boundaries:
        plt.axvline(x=b, color="orange", linestyle="--", alpha=0.7)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PPO Curriculum Training — Gazebo Task A")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/training_curve_gazebo.png", dpi=150)
    print("Training curve saved to logs/training_curve_gazebo.png")


if __name__ == "__main__":
    main()
