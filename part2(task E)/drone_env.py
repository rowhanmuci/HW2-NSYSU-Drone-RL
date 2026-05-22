#!/usr/bin/env python3
"""
drone_env.py
------------
Gymnasium environment for Task E: Multi-Waypoint Cruising.

The drone must visit all waypoints in order while minimizing time.
Extends the pattern from rl_fly_to_target.py (provided starter code).

Observation (13-D float32):
    [0:3]   drone position      (x, y, z)
    [3:6]   drone velocity      (vx, vy, vz)
    [6:9]   current waypoint    (wx, wy, wz)
    [9]     distance to current waypoint
    [10]    progress ratio      (waypoints done / total)
    [11:13] unit direction to waypoint in XY plane (dx, dy)

Action (3-D float32 in [-1, 1]):
    velocity command (vx, vy, vz) in m/s
"""

from typing import List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty

import gymnasium as gym
from gymnasium import spaces


# ── Waypoint presets ────────────────────────────────────────────────────────
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


# ── ROS 2 interface ──────────────────────────────────────────────────────────
class DroneROSInterface(Node):
    """
    Thin ROS 2 node: publishes velocity/reset/takeoff commands,
    subscribes to ground-truth pose.  Follows the same topic layout
    as the provided rl_fly_to_target.py starter.
    """

    def __init__(self):
        super().__init__("rl_waypoint_interface")
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
        """Reset drone then take off and fly back near origin.

        The Gazebo reset topic does not teleport the drone — it only brings it
        to ground level wherever it currently is.  After takeoff we therefore
        run a P-controller that steers the drone back to (0, 0, 2) before
        handing control back to the RL policy.  This prevents the cascade where
        one OOB episode leaves the drone permanently outside the flight zone.
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

        # P-controller: steer drone to hover at (0, 0, 2) before episode starts.
        # Uses proportional speed (faster when far) so it can recover from large OOB.
        home     = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        deadline = time.time() + 60.0   # 60 s: enough for up to ~60 m away
        while time.time() < deadline:
            err  = home - self.current_pose
            dist = float(np.linalg.norm(err))
            if dist < 1.0:
                break
            # Speed proportional to distance, capped at 1 m/s (cmd_vel limit)
            speed = min(1.0, dist * 0.5)
            vel   = (err / dist) * speed
            self.send_velocity(float(vel[0]), float(vel[1]), float(vel[2]))
            rclpy.spin_once(self, timeout_sec=0.1)

        self.send_velocity(0.0, 0.0, 0.0)
        t_end = time.time() + 1.0   # settle: let velocity estimate decay to ~0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.current_vel = np.zeros(3, dtype=np.float32)


# ── Gymnasium environment ────────────────────────────────────────────────────
class DroneWaypointEnv(gym.Env):
    """
    Task E – Multi-Waypoint Cruising.

    Episode ends when:
      - All waypoints are visited              (success, terminated=True)
      - Drone crashes (altitude < 0.1 m)       (failure, terminated=True)
      - Drone leaves the flight zone            (failure, terminated=True)
      - Step budget is exhausted                (truncated=True)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ros_interface: Optional[DroneROSInterface] = None,
        waypoints: List[Tuple[float, float, float]] = WAYPOINTS_MEDIUM,
        success_radius: float = 1.5,   # larger radius: easier to reach by chance during exploration
        max_steps: int = 300,
        out_of_bounds: float = 8.0,   # moderate: avoid walls (~6-8 m) but don't punish movement
    ):
        super().__init__()
        self._ros        = ros_interface
        self._waypoints  = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.success_radius = success_radius
        self.max_steps      = max_steps
        self.out_of_bounds  = out_of_bounds

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        self._current_wp_idx = 0
        self._step_count     = 0
        self._prev_dist      = 0.0

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

    @property
    def _current_wp(self) -> np.ndarray:
        return self._waypoints[min(self._current_wp_idx, len(self._waypoints) - 1)]

    def _get_obs(self) -> np.ndarray:
        ros  = self._ros_node
        pos  = ros.current_pose
        vel  = ros.current_vel
        wp   = self._current_wp
        diff = wp - pos
        dist = float(np.linalg.norm(diff))

        progress = float(self._current_wp_idx) / len(self._waypoints)

        # Unit vector in XY plane toward current waypoint
        horiz      = diff[:2]
        horiz_norm = float(np.linalg.norm(horiz))
        direction  = (horiz / horiz_norm).astype(np.float32) if horiz_norm > 1e-6 \
                     else np.zeros(2, dtype=np.float32)

        return np.concatenate(
            [pos, vel, wp, [dist], [progress], direction]
        ).astype(np.float32)

    # ── Gym API ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ros_node.reset_and_takeoff()
        self._current_wp_idx = 0
        self._step_count     = 0
        pos = self._ros_node.current_pose
        self._prev_dist = float(np.linalg.norm(pos - self._current_wp))
        return self._get_obs(), {}

    def step(self, action):
        ros    = self._ros_node
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ros.send_velocity(*action)
        rclpy.spin_once(ros, timeout_sec=0.1)
        self._step_count += 1

        pos  = ros.current_pose
        wp   = self._current_wp
        dist = float(np.linalg.norm(pos - wp))

        # ── reward ───────────────────────────────────────────────────────────
        # Potential-based shaping: reward for getting closer, penalty for drifting
        reward     = (self._prev_dist - dist) * 2.0
        reward    -= 0.01            # per-step time penalty
        self._prev_dist = dist
        terminated = False

        if dist < self.success_radius:
            reward += 20.0
            self._current_wp_idx += 1
            if self._current_wp_idx < len(self._waypoints):
                self._prev_dist = float(np.linalg.norm(
                    pos - self._current_wp))
            else:
                reward    += 100.0
                terminated = True

        # Grace period: ignore crash/OOB for first 20 steps (takeoff phase)
        if self._step_count > 20:
            if pos[2] < 0.1:            # crash / ground contact
                reward    -= 100.0
                terminated = True

            if (np.any(np.abs(pos[:2]) > self.out_of_bounds)
                    or pos[2] > self.out_of_bounds):
                reward    -= 50.0
                terminated = True

        truncated = self._step_count >= self.max_steps

        obs  = self._get_obs()
        info = {
            "waypoints_done":  self._current_wp_idx,
            "total_waypoints": len(self._waypoints),
            "distance_to_wp":  dist,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._ros is not None:
            self._ros.send_velocity(0.0, 0.0, 0.0)
