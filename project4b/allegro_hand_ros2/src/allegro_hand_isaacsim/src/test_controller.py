#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import time


class JointStepTest(Node):

    def __init__(self):
        super().__init__('allegro_hand_step_test')

        # Publisher
        self.joint_pub = self.create_publisher(
            JointState,
            '/allegroHand_0/joint_cmd',
            1
        )

        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState,
            '/allegroHand_0/joint_states',
            self.joint_callback,
            1
        )

        self.last_joint_state = None

        # Test parameters
        self.target_joint_index = 0
        self.target_position = 0.5

        # Phases
        self.zeroing = True
        self.zero_start_time = None
        self.zero_duration = 3.0

        self.step_sent = False
        self.step_time = None

        # Data recording
        self.time_data = []
        self.goal_data = []
        self.actual_data = []

        self.start_time = time.time()

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info("Step response test node started.")

    def joint_callback(self, msg: JointState):
        self.last_joint_state = msg

    def timer_callback(self):

        if self.last_joint_state is None:
            return

        now = time.time()

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(self.last_joint_state.name)
        joint_msg.position = [0.0] * len(joint_msg.name)

        # ------------------------
        # Phase 1: Zero hand
        # ------------------------
        if self.zeroing:

            if self.zero_start_time is None:
                self.zero_start_time = now
                self.get_logger().info("Zeroing hand...")

            self.joint_pub.publish(joint_msg)

            if now - self.zero_start_time > self.zero_duration:
                self.zeroing = False
                self.step_time = now
                self.get_logger().info("Sending step command")

            return

        # ------------------------
        # Phase 2: Step input
        # ------------------------
        joint_msg.position[self.target_joint_index] = self.target_position
        self.joint_pub.publish(joint_msg)

        # Record data
        t = now - self.step_time

        actual_position = self.last_joint_state.position[self.target_joint_index]

        self.time_data.append(t)
        self.goal_data.append(self.target_position)
        self.actual_data.append(actual_position)

        # Record for 5 seconds
        if t > 5.0:
            self.get_logger().info("Test finished. Plotting results.")
            self.plot_results()
            rclpy.shutdown()

    def plot_results(self):

        plt.figure()

        plt.plot(self.time_data, self.goal_data, label="Goal", linestyle="--")
        plt.plot(self.time_data, self.actual_data, label="Actual")

        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position (rad)")
        plt.title("Joint Step Response")

        plt.legend()
        plt.grid(True)

        plt.show()


def main(args=None):

    rclpy.init(args=args)

    node = JointStepTest()

    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()