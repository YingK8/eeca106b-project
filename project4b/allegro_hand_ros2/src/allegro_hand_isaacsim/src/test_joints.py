#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState 

class JointTestNode(Node):

    def __init__(self):
        super().__init__('allegro_hand_joint_test')

        # Parameters
        self.declare_parameter('amplitude', 0.5)
        self.declare_parameter('step_time', 2.0)

        self.amplitude = self.get_parameter('amplitude').value
        self.step_time = self.get_parameter('step_time').value

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

        self.current_joint_index = 0
        self.phase = 0  # 0 = push, 1 = return
        self.last_joint_state = None

        # Zeroing state
        self.zeroing_done = False
        self.zero_start_time = None
        self.zero_duration = 3.0  # seconds

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info("Joint testing node started. Waiting for joint states...")

    def joint_callback(self, msg: JointState):
        self.last_joint_state = msg

    def timer_callback(self):
        if self.last_joint_state is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ----------------------------
        # PHASE 1: Zero all joints
        # ----------------------------
        if not self.zeroing_done:
            if self.zero_start_time is None:
                self.zero_start_time = now
                self.get_logger().info("Zeroing all joints for 3 seconds...")

            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = list(self.last_joint_state.name)
            joint_msg.position = [0.0] * len(joint_msg.name)

            self.joint_pub.publish(joint_msg)

            if now - self.zero_start_time >= self.zero_duration:
                self.zeroing_done = True
                self.get_logger().info("Zeroing complete. Starting joint test.")
                # Reset timer period for normal operation
                self.timer.cancel()
                self.timer = self.create_timer(self.step_time, self.timer_callback)

            return

        # ----------------------------
        # PHASE 2: Normal test logic
        # ----------------------------
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(self.last_joint_state.name)
        joint_msg.position = [0.0] * len(joint_msg.name)

        if len(joint_msg.name) == 0:
            return

        if self.current_joint_index >= len(joint_msg.name):
            self.current_joint_index = 0

        if self.phase == 0:
            joint_msg.position[self.current_joint_index] += self.amplitude
            self.phase = 1
            self.get_logger().info(
                f"Pushing joint {joint_msg.name[self.current_joint_index]}"
            )
        else:
            self.phase = 0
            self.get_logger().info(
                f"Returning joint {joint_msg.name[self.current_joint_index]}"
            )
            self.current_joint_index += 1

        self.joint_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()