#!/usr/bin/env python3
"""
Controllers for Project 1 Part 1

Author: EECS 106B Course Staff, Spring 2026
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

class Controller:
    """
    Base controller class for UR7e robot.
    Subclasses should implement step_control() method.
    """

    def __init__(self, node):
        """
        Parameters
        ----------
        node : rclpy.node.Node
            ROS2 node for logging and publishing
        """
        self.node = node

    def step_control(self, target_position, target_velocity, current_position, current_velocity):
        """
        Calculate control input (joint velocities).

        Parameters
        ----------
        target_position : np.ndarray
            Desired joint positions (6-DOF for UR7e)
        target_velocity : np.ndarray
            Desired joint velocities (6-DOF)
        current_position : np.ndarray
            Current joint positions
        current_velocity : np.ndarray
            Current joint velocities

        Returns
        -------
        np.ndarray
            Commanded joint velocities (6-DOF)
        """
        raise NotImplementedError

    def get_name(self):
        """Returns the name of this controller"""
        raise NotImplementedError


class UR7eTrajectoryController:
    """
    Simple trajectory execution controller for UR7e.
    Uses the robot's built-in trajectory action server.
    """

    def __init__(self, node):
        """
        Parameters
        ----------
        node : rclpy.node.Node
            ROS2 node for logging and action client
        """
        self.node = node
        self.trajectory_client = ActionClient(
            node,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.node.get_logger().info("Initialized UR7e Trajectory Controller")

    def execute_joint_trajectory(self, joint_trajectory):
        """
        Execute a pre-computed joint trajectory using the robot's action server.

        Parameters
        ----------
        joint_trajectory : JointTrajectory
            Complete trajectory with waypoints and timestamps

        Returns
        -------
        bool
            True if trajectory execution succeeded, False otherwise
        """
        self.node.get_logger().info('Waiting for trajectory controller...')
        self.trajectory_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory

        self.node.get_logger().info('Sending trajectory to controller...')
        send_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('Trajectory rejected by controller')
            return False

        self.node.get_logger().info('Trajectory accepted, executing...')
        self.node.get_logger().info(f'Trajectory has {len(joint_trajectory.points)} waypoints')
        if len(joint_trajectory.points) > 0:
            last_time = joint_trajectory.points[-1].time_from_start.sec + \
                       joint_trajectory.points[-1].time_from_start.nanosec * 1e-9
            self.node.get_logger().info(f'Expected duration: {last_time:.2f} seconds')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        result = result_future.result()
        self.node.get_logger().info(f'Trajectory execution complete! Result: {result.result.error_code}')
        return True

    def get_name(self):
        """Returns the name of this controller"""
        return "UR7e Trajectory Controller"

class PIDJointVelocityController(Controller):
    """
    PID feedback controller for joint velocity control.
    Uses proportional, integral, and derivative gains to correct position and velocity errors.
    """

    def __init__(self, node, Kp, Ki, Kd):
        """
        Parameters
        ----------
        node : rclpy.node.Node
            ROS2 node for logging
        Kp : np.ndarray or list
            Proportional gains (6x1)
        Ki : np.ndarray or list
            Integral gains (6x1)
        Kd : np.ndarray or list
            Derivative gains (6x1)
        """
        super().__init__(node)
        # TODO: Add your code here! 
        self.Kp = np.diag(Kp)
        self.Ki = np.diag(Ki)
        self.Kd = np.diag(Kd)
        self.integral_err = np.zeros(6, dtype=float)

        self.node.get_logger().info("Initialized PID Joint Velocity Controller")

    def step_control(self, target_position, target_velocity, current_position, current_velocity):
        """
        Calculate control input using PID control.

        The control law:
            u = target_velocity + Kp * position_error + Ki * integral_error + Kd * velocity_error

        Where:
            position_error = target_position - current_position
            velocity_error = target_velocity - current_velocity
            integral_error is accumulated over time

        Parameters
        ----------
        target_position : np.ndarray
            Desired joint positions
        target_velocity : np.ndarray
            Desired joint velocities
        current_position : np.ndarray
            Current joint positions
        current_velocity : np.ndarray
            Current joint velocities

        Returns
        -------
        np.ndarray
            Commanded joint velocities
        """

        # TODO: Add your code here!
        position_error = target_position - current_position
        velocity_error = target_velocity - current_velocity

        u = target_velocity + (self.Kp @ position_error) + (self.Ki @ self.integral_err) + (self.Kd @ velocity_error)

        self.integral_err += position_error

        return u

    def get_name(self):
        """Returns the name of this controller"""
        return "PID Joint Velocity"


def main(args=None):

    rclpy.init(args=args)
    node = Node('controller_test')

    controller = UR7eTrajectoryController(node)

    node.get_logger().info(f"Created controller: {controller.get_name()}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
