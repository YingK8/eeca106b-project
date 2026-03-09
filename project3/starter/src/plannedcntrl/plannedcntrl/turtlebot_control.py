#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist, PointStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
import time
import sys
import os
import pickle
from dataclasses import dataclass, field

@dataclass
class PlannerResult:
    success: bool
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    omega: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: float = 0.0
    total_time: float = 0.0
    solver_stats: dict = field(default_factory=dict)

class TurtleBotController(Node):
    def __init__(self, trajectory_filename=None):
        super().__init__('turtlebot_controller')
        
        self.trajectory_filename = trajectory_filename
        self.mission_started = False

        # Publisher and TF setup
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to trigger planning once pose is available
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info('TurtleBot controller node initialized.')

    def get_current_pose(self):
        try:
            # Get robot pose (base_link) in odom frame
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = quaternion_to_yaw(q.w, q.x, q.y, q.z)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")
            return None

    def timer_callback(self):
        if self.mission_started:
            return
        # Timer stops so we don't plan again
        self.timer.cancel()
        self.mission_started = True

        self.plan_and_follow()

    def plan_and_follow(self):
        if self.trajectory_filename is None:
            self.get_logger().info("No trajectory file provided.")
            return
        trajectory_data = np.load(self.trajectory_filename, allow_pickle=True)
        result = PlannerResult(
            x=trajectory_data['x'],
            y=trajectory_data['y'],
            theta=trajectory_data['theta'],
            v=trajectory_data['v'],
            omega=trajectory_data['omega'],
            total_time=trajectory_data['total_time'],
            dt=trajectory_data['dt'],
            solver_stats=None,
            success=True
        )
        self.follow_trajectory(result)

    def follow_trajectory(self, result):
        """
        Follow the generated trajectory by executing the computed controls (v, omega).
        """
        ### TODO: Implement trajectory following code here ###
            
        # Stop the robot after trajectory is done
        self.pub.publish(Twist())
        self.get_logger().info("Trajectory finished.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_from_yaw(yaw):
        """Return quaternion (x, y, z, w) from yaw angle."""
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    @staticmethod
    def quaternion_to_yaw(w, x, y, z):
        """
        Convert a quaternion (w, x, y, z) into the yaw angle (rotation around z-axis).
        The result is in radians.
        """
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return yaw


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    
    # You can change the file name to load different trajectories
    trajectory_filename = ".../optimization_trajectory_cory105_min_time.npz"

    node = TurtleBotController(trajectory_filename=trajectory_filename)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
