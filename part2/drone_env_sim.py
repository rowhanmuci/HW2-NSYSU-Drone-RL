#!/usr/bin/env python3
"""
drone_env_sim.py
----------------
Lightweight local simulation for Task A: Precision Hovering.
No ROS 2 or Gazebo needed — runs at ~50,000 steps/sec.

Identical observation space (10-D), action space (3-D ±1), and reward
function to DroneHoverEnv, so a policy trained here can be loaded
directly in the real ROS 2 environment for evaluation.

Physics: first-order velocity lag (TAU = 0.2 s) approximating the
real drone's response to cmd_vel commands.
"""

from typing import Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_TARGET: Tuple[float, float, float] = (2.0, 0.0, 2.0)


class DroneHoverSimEnv(gym.Env):
    """
    Fast drone simulation for rapid PPO training on hovering.

    Each episode starts at a random position within ±1 m of the origin
    (height 1–3 m). The goal is to reach TARGET and maintain a stable
    hover (dist < success_radius) for stable_steps_required consecutive steps.
    Velocity is penalised per-step but not checked for the success condition,
    so noise-driven corrections cannot prevent success from triggering.

    Physics: vel = (1-alpha)*vel + alpha*action  then  pos += vel*DT.

    Observation (10-D float32):
        [0:3]  pos  (x, y, z)
        [3:6]  vel  (vx, vy, vz)
        [6:9]  target (tx, ty, tz)
        [9]    distance to target

    Action (3-D float32 in [-1, 1]):
        velocity command (vx, vy, vz) in m/s
    """

    metadata = {"render_modes": []}

    DT = 0.1  # seconds per step (matches real env ~10 Hz)

    def __init__(
        self,
        target: Tuple[float, float, float] = DEFAULT_TARGET,
        noise_level: float = 0.0,
        success_radius: float = 0.3,
        stable_steps_required: int = 30,
        max_steps: int = 500,
        out_of_bounds: float = 6.0,
        tau_range: Tuple[float, float] = (0.05, 0.4),
    ):
        super().__init__()
        self._target               = np.array(target, dtype=np.float32)
        self.noise_level           = noise_level
        self.success_radius        = success_radius
        self.stable_steps_required = stable_steps_required
        self.max_steps             = max_steps
        self.out_of_bounds         = out_of_bounds
        self.tau_range             = tau_range   # (min, max) randomised each episode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self._pos          = np.zeros(3, dtype=np.float32)
        self._vel          = np.zeros(3, dtype=np.float32)
        self._steps        = 0
        self._stable_count = 0
        self._prev_dist    = 0.0
        self._alpha        = self.DT / 0.2   # initialised; overwritten in reset()

    def _get_obs(self) -> np.ndarray:
        dist = float(np.linalg.norm(self._pos - self._target))
        return np.concatenate(
            [self._pos, self._vel, self._target, [dist]]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # Domain randomisation: sample a new TAU each episode so the policy
        # learns to handle a range of velocity-lag dynamics.  Gazebo's real
        # TAU is unknown but likely falls somewhere in tau_range.
        tau          = rng.uniform(self.tau_range[0], self.tau_range[1])
        self._alpha  = float(self.DT / tau)
        self._pos = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(1.0,  3.0),
        ], dtype=np.float32)
        self._vel          = np.zeros(3, dtype=np.float32)
        self._steps        = 0
        self._stable_count = 0
        self._prev_dist    = float(np.linalg.norm(self._pos - self._target))
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # First-order lag: velocity tracks command with time constant TAU
        self._vel = (1.0 - self._alpha) * self._vel + self._alpha * action
        self._pos = self._pos + self._vel * self.DT

        # Apply drift disturbance (simulates motionDriftNoise / wind)
        if self.noise_level > 0.0:
            drift = self.np_random.standard_normal(3).astype(np.float32)
            self._pos += drift * self.noise_level * 0.05
            self._pos[2] = max(float(self._pos[2]), 0.05)

        self._steps += 1

        dist    = float(np.linalg.norm(self._pos - self._target))
        vel_mag = float(np.linalg.norm(self._vel))

        # ── reward ─────────────────────────────────────────────────────────────
        # Potential-based shaping: positive when moving toward target.
        # Gives a non-zero advantage difference between "move" and "stay"
        # even when all absolute rewards are negative.
        shaping = (self._prev_dist - dist) * 10.0
        self._prev_dist = dist

        reward  = -dist * 2.0       # dense distance penalty
        reward += shaping           # navigation gradient
        reward -= vel_mag * 0.05    # small velocity penalty (stability)
        reward -= 0.01              # per-step time penalty

        terminated = False

        # Stability tracking: position-only so noise-driven velocity corrections
        # don't block success. Velocity is still penalised in the per-step reward.
        if dist < self.success_radius:
            self._stable_count += 1
            if self._stable_count >= self.stable_steps_required:
                reward    += 100.0
                terminated = True
        else:
            self._stable_count = 0

        # Crash (ignore first 10 steps — takeoff grace period)
        if self._pos[2] < 0.1 and self._steps > 10:
            reward    -= 100.0
            terminated = True

        # Out of bounds
        if (np.any(np.abs(self._pos) > self.out_of_bounds)
                or self._pos[2] > self.out_of_bounds):
            reward    -= 50.0
            terminated = True

        truncated = self._steps >= self.max_steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {
                "distance":    dist,
                "vel_mag":     vel_mag,
                "stable_count": self._stable_count,
                "success":     terminated and self._stable_count >= self.stable_steps_required,
            },
        )
