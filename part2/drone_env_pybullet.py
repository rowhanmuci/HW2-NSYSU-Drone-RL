#!/usr/bin/env python3
"""
drone_env_pybullet.py
---------------------
DroneHoverPybulletEnv: Task A wrapper around gym-pybullet-drones.

Calibrated to match the assignment's Gazebo simple_drone as closely as
possible without changing the CF2X URDF:

  • SPEED_LIMIT overridden to 1.0 m/s  (Gazebo cmd_vel effective range)
  • Drone teleported to (0, 0, 2) ± small noise after each reset
    so the start position matches Gazebo's post-takeoff hover point
  • max_steps = 300  (same as DroneHoverEnv)

The remaining gap (Crazyflie mass/inertia vs simple_drone) is bridged
partly by domain randomisation on the start position and target offset,
so the policy learns to be robust to small dynamic differences.

Observation (10-D, identical to drone_env_sim.py / drone_env.py):
    [0:3]  pos    (x, y, z)    m
    [3:6]  vel    (vx, vy, vz) m/s
    [6:9]  target (tx, ty, tz) m
    [9]    dist                m

Action (3-D in [-1, 1]):
    Velocity command → pybullet VEL action [dir_x, dir_y, dir_z, speed_frac].
    With SPEED_LIMIT = 1.0 m/s, action magnitude directly equals commanded
    speed in m/s, matching the Gazebo cmd_vel convention.

Usage
-----
    python drone_env_pybullet.py                        # sanity check
    python train.py --pybullet --n-envs 4 --noise 0.1  # training
    python test.py  --pybullet                          # evaluation
    python test.py  --model models/ppo_hover_pb_final   # Gazebo deploy (no scale needed)
"""

from typing import Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_TARGET: Tuple[float, float, float] = (2.0, 0.0, 2.0)


class DroneHoverPybulletEnv(gym.Env):
    """
    Task A – Precision Hovering with pybullet Crazyflie physics,
    calibrated to match Gazebo simple_drone conditions.

    Action mapping
    --------------
    Our 3-D action ∈ [-1, 1]^3 is normalised to a direction vector and
    its magnitude becomes the speed_frac for pybullet's VEL controller:

        commanded_vel = SPEED_LIMIT * |action| * normalize(action)

    SPEED_LIMIT is forced to 1.0 m/s so action=1.0 ≈ 1 m/s, matching
    the Gazebo cmd_vel convention (no action-scale needed at deploy time).

    Start / target randomisation
    ----------------------------
    After each reset the drone is teleported to a random position within
    ±0.3 m (XY) and ±0.3 m (Z) of (0, 0, 2) — Gazebo's post-takeoff
    hover point.  The target is additionally perturbed ±target_rand_range
    around the base target for generalisation.

    Parallelisation note
    --------------------
    Each env instance owns one pybullet client.  Use SubprocVecEnv (not
    DummyVecEnv) so that clients stay isolated in separate processes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target: Tuple[float, float, float] = DEFAULT_TARGET,
        noise_level: float = 0.0,
        success_radius: float = 0.3,
        stable_steps_required: int = 30,
        max_steps: int = 300,
        out_of_bounds: float = 6.0,
        gui: bool = False,
        target_rand_range: float = 0.5,
    ):
        super().__init__()
        self._target_base          = np.array(target, dtype=np.float32)
        self.noise_level           = noise_level
        self.success_radius        = success_radius
        self.stable_steps_required = stable_steps_required
        self.max_steps             = max_steps
        self.out_of_bounds         = out_of_bounds
        self.gui                   = gui
        self.target_rand_range     = target_rand_range

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Aviary is lazily created on first reset() so pybullet starts
        # inside the correct process when using SubprocVecEnv.
        self._aviary       = None
        self._speed_limit  = 0.25          # overwritten after aviary creation
        self._target       = self._target_base.copy()
        self._pos          = np.zeros(3, dtype=np.float32)
        self._vel          = np.zeros(3, dtype=np.float32)
        self._steps        = 0
        self._stable_count = 0
        self._prev_dist    = 0.0

    # ── pybullet helpers ──────────────────────────────────────────────────────

    # Target speed that matches Gazebo cmd_vel effective range (~1 m/s).
    # The CF2X default is 0.25 m/s; we override it after aviary creation.
    GAZEBO_SPEED_LIMIT: float = 1.0

    def _init_aviary(self) -> None:
        from gym_pybullet_drones.envs import HoverAviary
        from gym_pybullet_drones.utils.enums import ObservationType, ActionType

        self._aviary = HoverAviary(
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            gui=self.gui,
            record=False,
        )
        # Override the default 0.25 m/s to match Gazebo's effective speed.
        self._aviary.SPEED_LIMIT = self.GAZEBO_SPEED_LIMIT
        self._speed_limit = self.GAZEBO_SPEED_LIMIT

    def _teleport_to(self, pos: np.ndarray) -> None:
        """Move drone to pos and zero all velocities via pybullet API."""
        import pybullet as p
        drone_id = self._aviary.DRONE_IDS[0]
        client   = self._aviary.CLIENT
        p.resetBasePositionAndOrientation(
            drone_id,
            [float(pos[0]), float(pos[1]), float(pos[2])],
            [0., 0., 0., 1.],
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            drone_id,
            [0., 0., 0.],
            [0., 0., 0.],
            physicsClientId=client,
        )
        self._pos = pos.astype(np.float32)
        self._vel = np.zeros(3, dtype=np.float32)

    def _read_state(self, raw_obs: np.ndarray) -> None:
        """Extract pos [0:3] and vel [6:9] from the 72-D KIN obs vector.

        KIN layout per drone: [pos(3), euler(3), vel(3), ang_vel(3)] = 12,
        then action history is appended.  Indices are stable across versions.
        """
        self._pos = raw_obs[0, 0:3].astype(np.float32)
        self._vel = raw_obs[0, 6:9].astype(np.float32)

    def _to_pb_action(self, action: np.ndarray) -> np.ndarray:
        """Convert 3-D velocity command to pybullet 4-D VEL action."""
        speed = float(np.linalg.norm(action))
        if speed > 1e-6:
            direction = action / speed
        else:
            direction = np.zeros(3, dtype=np.float32)
        speed_frac = float(np.clip(speed, 0.0, 1.0))
        return np.array(
            [[direction[0], direction[1], direction[2], speed_frac]],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        dist = float(np.linalg.norm(self._pos - self._target))
        return np.concatenate(
            [self._pos, self._vel, self._target, [dist]]
        ).astype(np.float32)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        if self._aviary is None:
            self._init_aviary()

        self._aviary.reset()

        # Teleport drone to a random start near (0, 0, 2) — Gazebo's
        # post-takeoff hover point.  Small XY/Z noise prevents memorisation.
        start = np.array([
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            rng.uniform(1.7, 2.3),
        ], dtype=np.float32)
        self._teleport_to(start)

        # Randomise target around the base target for generalisation.
        offset = np.array([
            rng.uniform(-self.target_rand_range, self.target_rand_range),
            rng.uniform(-self.target_rand_range, self.target_rand_range),
            rng.uniform(-self.target_rand_range / 2, self.target_rand_range / 2),
        ], dtype=np.float32)
        self._target = np.clip(
            self._target_base + offset,
            np.array([-5.5, -5.5, 0.5]),
            np.array([ 5.5,  5.5, 5.5]),
        )

        self._steps        = 0
        self._stable_count = 0
        self._prev_dist    = float(np.linalg.norm(self._pos - self._target))
        return self._get_obs(), {}

    def step(self, action):
        action    = np.clip(action, -1.0, 1.0).astype(np.float32)
        pb_action = self._to_pb_action(action)

        # Ignore pybullet's own termination: its EPISODE_LEN_SEC and hover-at-
        # (0,0,1) logic don't match our task.  We handle all done conditions below.
        raw_obs, _, _pb_term, _pb_trunc, _ = self._aviary.step(pb_action)
        self._read_state(raw_obs)

        # Optional drift noise to simulate motionDriftNoise
        if self.noise_level > 0.0:
            drift         = self.np_random.standard_normal(3).astype(np.float32)
            self._pos    += drift * self.noise_level * 0.05
            self._pos[2]  = max(float(self._pos[2]), 0.05)

        self._steps += 1
        dist    = float(np.linalg.norm(self._pos - self._target))
        vel_mag = float(np.linalg.norm(self._vel))

        # ── reward ─────────────────────────────────────────────────────────────
        shaping = (self._prev_dist - dist) * 10.0
        self._prev_dist = dist

        reward  = -dist * 2.0 + shaping - vel_mag * 0.05 - 0.01
        terminated = False

        if dist < self.success_radius:
            self._stable_count += 1
            if self._stable_count >= self.stable_steps_required:
                reward    += 100.0
                terminated = True
        else:
            self._stable_count = 0

        if self._pos[2] < 0.1 and self._steps > 10:
            reward    -= 100.0
            terminated = True

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
                "distance":     dist,
                "vel_mag":      vel_mag,
                "stable_count": self._stable_count,
                "success":      terminated and self._stable_count >= self.stable_steps_required,
            },
        )

    def close(self) -> None:
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Creating environment …")
    env = DroneHoverPybulletEnv(gui=False)
    obs, _ = env.reset()
    print(f"  Start pos:  {obs[0:3]}")
    print(f"  Target:     {obs[6:9]}")
    print(f"  Dist:       {obs[9]:.3f} m")
    print(f"  SPEED_LIMIT: {env._speed_limit:.3f} m/s")

    # Fly toward the target with a simple proportional controller
    t0 = time.time()
    total_rew = 0.0
    for step in range(300):
        error     = obs[6:9] - obs[0:3]          # target - pos
        dist_err  = np.linalg.norm(error)
        speed_cmd = np.clip(dist_err, 0.0, 1.0)
        action    = (error / (dist_err + 1e-6)) * speed_cmd
        obs, rew, term, trunc, info = env.step(action)
        total_rew += rew
        if term or trunc:
            break

    elapsed = time.time() - t0
    print(f"\n  Steps: {step + 1}   Reward: {total_rew:.1f}")
    print(f"  Final dist: {info['distance']:.3f} m   "
          f"Success: {info['success']}   "
          f"({elapsed:.1f} s elapsed)")
    env.close()
