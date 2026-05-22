#!/usr/bin/env python3
"""
drone_env.py
------------
Gymnasium environment for Task A: Precision Hovering.

The drone must reach a fixed target (x, y, z) and maintain a stable
hover — position error < success_radius AND |velocity| < 0.3 m/s —
for stable_steps_required consecutive steps, even under wind disturbance
(motionDriftNoise in Gazebo).

Observation (10-D float32):
    [0:3]  drone position      (x, y, z)
    [3:6]  drone velocity      (vx, vy, vz)
    [6:9]  hover target        (tx, ty, tz)
    [9]    distance to target

Action (3-D float32 in [-1, 1]):
    velocity command (vx, vy, vz) in m/s
"""

from typing import Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty

import gymnasium as gym
from gymnasium import spaces

DEFAULT_TARGET: Tuple[float, float, float] = (2.0, 0.0, 2.0)


# ── ROS 2 interface ──────────────────────────────────────────────────────────
class DroneROSInterface(Node):
    """
    Thin ROS 2 node: publishes velocity/reset/takeoff commands,
    subscribes to ground-truth pose.  Topic layout matches the
    provided rl_fly_to_target.py starter and Task E.
    """

    def __init__(self):
        super().__init__("rl_hover_interface")
        self.current_pose = np.zeros(3, dtype=np.float32)
        self.current_vel  = np.zeros(3, dtype=np.float32)

        self._cmd_pub     = self.create_publisher(Twist, "/simple_drone/cmd_vel",  10)
        self._takeoff_pub = self.create_publisher(Empty, "/simple_drone/takeoff",  10)
        self._reset_pub   = self.create_publisher(Empty, "/simple_drone/reset",    10)
        self.create_subscription(Pose, "/simple_drone/gt_pose", self._pose_cb, 10)

    def _pose_cb(self, msg: Pose) -> None:
        new_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z], dtype=np.float32
        )
        # Simple finite-difference velocity estimate (~10 Hz callback)
        self.current_vel  = (new_pose - self.current_pose) * 10.0
        self.current_pose = new_pose

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self._cmd_pub.publish(msg)

    def reset_and_takeoff(self) -> None:
        """Reset drone, take off, then fly back to the origin hover point.

        Gazebo's reset topic lands the drone at its current location.
        A P-controller then steers it to (0, 0, 2) before handing
        control to the RL policy, preventing permanent OOB episodes.
        """
        import time
        self.send_velocity(0.0, 0.0, 0.0)
        t_end = time.time() + 0.5
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)

        self._reset_pub.publish(Empty())
        t_end = time.time() + 2.0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)

        self._takeoff_pub.publish(Empty())
        t_end = time.time() + 2.5
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)

        # P-controller: return drone to (0, 0, 2) before episode starts
        home     = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        deadline = time.time() + 60.0
        while time.time() < deadline:
            err  = home - self.current_pose
            dist = float(np.linalg.norm(err))
            if dist < 0.3:
                break
            speed = min(1.0, dist * 0.5)
            vel   = (err / dist) * speed
            self.send_velocity(float(vel[0]), float(vel[1]), float(vel[2]))
            rclpy.spin_once(self, timeout_sec=0.1)

        self.send_velocity(0.0, 0.0, 0.0)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.current_vel = np.zeros(3, dtype=np.float32)


# ── Gymnasium environment ────────────────────────────────────────────────────
class DroneHoverEnv(gym.Env):
    """
    Task A – Precision Hovering.

    Episode ends when:
      - Stable hover achieved (dist < radius, |vel| < 0.3 for N steps)  (success, terminated=True)
      - Drone crashes (altitude < 0.1 m)                                (failure, terminated=True)
      - Drone leaves the flight zone                                     (failure, terminated=True)
      - Step budget is exhausted                                         (truncated=True)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ros_interface: Optional[DroneROSInterface] = None,
        target: Tuple[float, float, float] = DEFAULT_TARGET,
        success_radius: float = 0.3,
        stable_steps_required: int = 30,
        max_steps: int = 300,
        out_of_bounds: float = 6.0,
    ):
        super().__init__()
        self._ros                  = ros_interface
        self._target               = np.array(target, dtype=np.float32)
        self.success_radius        = success_radius
        self.stable_steps_required = stable_steps_required
        self.max_steps             = max_steps
        self.out_of_bounds         = out_of_bounds

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self._step_count   = 0
        self._stable_count = 0
        self._prev_dist    = 0.0

    # ── helpers ─────────────────────────────────────────────────────────────
    def set_ros_interface(self, ros: DroneROSInterface) -> None:
        self._ros = ros

    @property
    def _ros_node(self) -> DroneROSInterface:
        if self._ros is None:
            raise RuntimeError(
                "No ROS interface attached. "
                "Pass ros_interface= to __init__ or call set_ros_interface()."
            )
        return self._ros

    def _get_obs(self) -> np.ndarray:
        pos  = self._ros_node.current_pose
        vel  = self._ros_node.current_vel
        dist = float(np.linalg.norm(pos - self._target))
        return np.concatenate(
            [pos, vel, self._target, [dist]]
        ).astype(np.float32)

    # ── Gym API ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ros_node.reset_and_takeoff()
        self._step_count   = 0
        self._stable_count = 0
        pos = self._ros_node.current_pose
        self._prev_dist = float(np.linalg.norm(pos - self._target))
        return self._get_obs(), {}

    def step(self, action):
        ros    = self._ros_node
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ros.send_velocity(*action)
        rclpy.spin_once(ros, timeout_sec=0.1)
        self._step_count += 1

        pos     = ros.current_pose
        vel     = ros.current_vel
        dist    = float(np.linalg.norm(pos - self._target))
        vel_mag = float(np.linalg.norm(vel))

        # ── reward ───────────────────────────────────────────────────────────
        # Large shaping coefficient (100 vs old 10) is the key fix:
        # Gazebo's physical damping means actual drone displacement is ~0.05 m/step.
        # With shaping×10 the per-episode return std is ~9, below V(s) estimation
        # noise (~20-50) → PPO cannot distinguish navigation from hovering.
        # With shaping×100 the std rises to ~90, well above noise → clear signal.
        #
        # Reduced distance penalty (0.5 vs old 2.0) keeps the negative baseline
        # from dominating, so the advantage signal from shaping is proportionally
        # larger relative to V(s).
        shaping = (self._prev_dist - dist) * 100.0
        self._prev_dist = dist

        reward  = -dist * 0.5       # gentle distance penalty (not dominant)
        reward += shaping           # strong navigation gradient
        reward -= vel_mag * 0.05    # velocity penalty (stability)
        reward -= 0.01              # per-step time penalty

        terminated = False

        if dist < self.success_radius:
            reward += 3.0          # per-step zone bonus: breaks zero-advantage loop
            self._stable_count += 1
            if self._stable_count >= self.stable_steps_required:
                reward    += 100.0
                terminated = True
        else:
            self._stable_count = 0

        # Grace period: ignore crash/OOB for first 20 steps (takeoff phase)
        if self._step_count > 20:
            if pos[2] < 0.1:
                reward    -= 100.0
                terminated = True

            if (np.any(np.abs(pos) > self.out_of_bounds)
                    or pos[2] > self.out_of_bounds):
                reward    -= 50.0
                terminated = True

        truncated = self._step_count >= self.max_steps

        success = terminated and self._stable_count >= self.stable_steps_required
        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {
                "distance":     dist,
                "vel_mag":      vel_mag,
                "stable_count": self._stable_count,
                "success":      success,
            },
        )

    def close(self) -> None:
        if self._ros is not None:
            self._ros.send_velocity(0.0, 0.0, 0.0)
