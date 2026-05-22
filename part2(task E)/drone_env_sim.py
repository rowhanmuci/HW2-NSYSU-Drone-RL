#!/usr/bin/env python3
"""
drone_env_sim.py
----------------
Lightweight local simulation of Task E (Multi-Waypoint Cruising).
No ROS 2 or Gazebo needed — runs at ~50,000 steps/sec.

Identical observation space (13-D), action space (3-D ±1), and reward
function to DroneWaypointEnv, so a policy trained here can be loaded
directly in the real ROS 2 environment for evaluation.

Physics: first-order velocity lag (TAU = 0.2 s) approximating the
real drone's response to cmd_vel commands.
"""

from typing import List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WAYPOINTS_EASY: List[Tuple[float, float, float]] = [
    (2.0,  0.0, 2.0),
    (4.0,  2.0, 2.0),
    (2.0,  4.0, 2.0),
    (0.0,  2.0, 2.0),
]

WAYPOINTS_MEDIUM: List[Tuple[float, float, float]] = [
    ( 3.0,  0.0, 2.0),
    ( 3.0,  3.0, 3.0),
    ( 0.0,  3.0, 2.5),
    (-2.0,  0.0, 2.0),
    ( 0.0, -2.0, 1.5),
]


class DroneWaypointSimEnv(gym.Env):
    """
    Fast drone simulation for rapid PPO training.

    Episode starts at (0, 0, 2) hovering at origin.
    Physics: vel = (1-alpha)*vel + alpha*action  then  pos += vel*DT.
    """

    metadata = {"render_modes": []}

    DT  = 0.1   # seconds per step (same as real env ~10 Hz)
    TAU = 0.2   # velocity lag time constant in seconds

    def __init__(
        self,
        waypoints: List[Tuple[float, float, float]] = WAYPOINTS_MEDIUM,
        success_radius: float = 0.8,
        max_steps: int = 500,
        out_of_bounds: float = 10.0,
    ):
        super().__init__()
        self._waypoints     = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.success_radius = success_radius
        self.max_steps      = max_steps
        self.out_of_bounds  = out_of_bounds

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        self._pos       = np.zeros(3, dtype=np.float32)
        self._vel       = np.zeros(3, dtype=np.float32)
        self._wp_idx    = 0
        self._steps     = 0
        self._prev_dist = 0.0

    @property
    def _current_wp(self) -> np.ndarray:
        return self._waypoints[min(self._wp_idx, len(self._waypoints) - 1)]

    def _get_obs(self) -> np.ndarray:
        wp   = self._current_wp
        diff = wp - self._pos
        dist = float(np.linalg.norm(diff))
        progress = float(self._wp_idx) / len(self._waypoints)

        horiz      = diff[:2]
        horiz_norm = float(np.linalg.norm(horiz))
        direction  = (horiz / horiz_norm).astype(np.float32) if horiz_norm > 1e-6 \
                     else np.zeros(2, dtype=np.float32)

        return np.concatenate(
            [self._pos, self._vel, wp, [dist], [progress], direction]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos    = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self._vel    = np.zeros(3, dtype=np.float32)
        self._wp_idx = 0
        self._steps  = 0
        self._prev_dist = float(np.linalg.norm(self._pos - self._current_wp))
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # First-order lag: velocity tracks command with time constant TAU
        alpha     = self.DT / self.TAU
        self._vel = (1.0 - alpha) * self._vel + alpha * action
        self._pos = self._pos + self._vel * self.DT
        self._steps += 1

        wp   = self._current_wp
        dist = float(np.linalg.norm(self._pos - wp))

        # Potential-based shaping (identical to DroneWaypointEnv)
        reward = (self._prev_dist - dist) * 2.0
        reward -= 0.01
        self._prev_dist = dist
        terminated = False

        if dist < self.success_radius:
            reward += 20.0
            self._wp_idx += 1
            if self._wp_idx < len(self._waypoints):
                self._prev_dist = float(np.linalg.norm(self._pos - self._current_wp))
            else:
                reward    += 100.0
                terminated = True

        if self._pos[2] < 0.1:
            reward    -= 100.0
            terminated = True

        if (np.any(np.abs(self._pos[:2]) > self.out_of_bounds)
                or self._pos[2] > self.out_of_bounds):
            reward    -= 50.0
            terminated = True

        truncated = self._steps >= self.max_steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {"waypoints_done":  self._wp_idx,
             "total_waypoints": len(self._waypoints),
             "distance_to_wp":  dist},
        )
