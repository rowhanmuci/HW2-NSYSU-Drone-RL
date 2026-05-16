#!/usr/bin/env python3
"""
fly_straight_part1.py
---------------------
Modified fly_straight.py for Part 1 testing.
Accepts target coordinates and controller parameters as CLI args
so all three alternative targets can be tested without editing the file.

Usage examples (inside Docker container):
    # Default target
    python3 fly_straight_part1.py

    # Alternative targets required by the assignment
    python3 fly_straight_part1.py --target -2 4 1.5
    python3 fly_straight_part1.py --target 0 0 3
    python3 fly_straight_part1.py --target 6 -3 2

    # Vary Kp to observe oscillation / sluggishness
    python3 fly_straight_part1.py --target 5 3 2 --kp 0.1
    python3 fly_straight_part1.py --target 5 3 2 --kp 1.5

    # Vary max_speed to observe overshooting
    python3 fly_straight_part1.py --target 5 3 2 --max_speed 0.3
    python3 fly_straight_part1.py --target 5 3 2 --max_speed 2.0
"""

import argparse
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty


class FlyToTarget(Node):
    def __init__(self, target_x, target_y, target_z, kp, max_speed, tolerance):
        super().__init__("fly_straight_part1")

        self.target   = (target_x, target_y, target_z)
        self.kp       = kp
        self.max_speed = max_speed
        self.tolerance = tolerance

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.reached   = False

        self.cmd_pub = self.create_publisher(Twist,  "/simple_drone/cmd_vel", 10)
        self.to_pub  = self.create_publisher(Empty,  "/simple_drone/takeoff", 10)
        self.create_subscription(Pose, "/simple_drone/gt_pose", self.pose_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info(
            f"Target: {self.target}  Kp={self.kp}  max_speed={self.max_speed}"
        )
        self.to_pub.publish(Empty())

    def pose_cb(self, msg: Pose) -> None:
        self.current_x = msg.position.x
        self.current_y = msg.position.y
        self.current_z = msg.position.z

    def control_loop(self) -> None:
        if self.reached:
            return

        tx, ty, tz = self.target
        ex = tx - self.current_x
        ey = ty - self.current_y
        ez = tz - self.current_z
        dist = math.sqrt(ex**2 + ey**2 + ez**2)

        if dist < self.tolerance:
            self.reached = True
            self.cmd_pub.publish(Twist())   # stop
            self.get_logger().info(
                f"Reached target {self.target}! "
                f"pos=({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})"
            )
            return

        # Proportional velocity command, clamped to max_speed
        vx = self.kp * ex
        vy = self.kp * ey
        vz = self.kp * ez
        scale = min(1.0, self.max_speed / (math.sqrt(vx**2 + vy**2 + vz**2) + 1e-6))
        vx, vy, vz = vx * scale, vy * scale, vz * scale

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz
        self.cmd_pub.publish(msg)

        self.get_logger().info(
            f"dist={dist:.2f}  pos=({self.current_x:.2f},{self.current_y:.2f},{self.current_z:.2f})"
            f"  vel=({vx:.2f},{vy:.2f},{vz:.2f})",
            throttle_duration_sec=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",    nargs=3, type=float, default=[5.0, 3.0, 2.0],
                        metavar=("X", "Y", "Z"), help="Target position (default: 5 3 2)")
    parser.add_argument("--kp",        type=float, default=0.5,
                        help="Proportional gain (default: 0.5)")
    parser.add_argument("--max_speed", type=float, default=1.0,
                        help="Maximum speed in m/s (default: 1.0)")
    parser.add_argument("--tolerance", type=float, default=0.2,
                        help="Arrival tolerance in m (default: 0.2)")
    args = parser.parse_args()

    rclpy.init()
    node = FlyToTarget(*args.target, args.kp, args.max_speed, args.tolerance)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
