#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import torch
import numpy as np
from policy_wrapper_real_world import (RslRlPolicyWrapperRealWorld, generate_goal_pose, goal_quat_diff,
                                        real2sim_joints, quat_mul, quat_conjugate)

class TestPolicyNode(Node):

    def __init__(self):
        super().__init__('test_policy_node')

        # Parameters
        self.declare_parameter('policy_path', '/home/cc/ee106b/sp26/class/ee106b-aau/106b-sp26-labs-starter/project4b/allegro_reorientation/logs/rsl_rl/exported/exported_checkpoint_task42.pt')
        self.declare_parameter('orientation_success_threshold', 1.0) # radians
        self.declare_parameter('action_scale', 1.0)

        self.policy_path = self.get_parameter('policy_path').value

        # Load Policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = self.create_policy()

        self.get_logger().info(f"Successfully loaded policy from {self.policy_path}")
        self.get_logger().info(f"Running on device: {self.device}")

        self.last_joint_state = None
        self.last_raw_action = np.zeros(16, dtype=np.float32)
        self.last_cube_pose = None
        self.initialized_buffer = False
        self._prev_cube_pos = None
        self._prev_cube_quat = None
        self._prev_cube_time = None
        self.orientation_success_threshold = self.get_parameter('orientation_success_threshold').value
        self.goal_pos, self.goal_quat = generate_goal_pose()

        # ROS Interfaces
        self.joint_pub = self.create_publisher(
            JointState,
            '/allegroHand_0/joint_cmd',
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/allegroHand_0/joint_states',
            self.joint_callback,
            10
        )

        self.cube_sub = self.create_subscription(
            PoseStamped, 
            '/cube_pose', 
            self.pose_callback, 
            10        
        )

        # TODO: if error below some threshold, resample goal pos, goal quat

        self.get_logger().info("Resetting hand at startup...")
        while self.last_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.send_zero_command()

        self.get_logger().info("Sleeping 2 seconds after zeroing...")

        start_time = self.get_clock().now()
        duration = rclpy.duration.Duration(seconds=2.0)

        while self.get_clock().now() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 30 Hz Control Loop
        self.control_rate = 30.0
        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info(f"Test policy node started ({self.control_rate} Hz).")
    
    def create_policy(self):
         # TODO: define q_min and q_max from the simulation policy setup before creating the real-world wrapper.
        # q_min = ...
        # q_max = ...
        # Joint limits in simulation order (from infer_allegro_repose.py reference script)
        # Order: index(0-3), middle(4-7), ring(8-11), thumb(12-15) in sim convention
        q_min = np.array([
            -0.5585, -0.5585, -0.5585,  0.2792,   # index
            -0.2792, -0.2792, -0.2792, -0.3316,   # middle
            -0.2792, -0.2792, -0.2792, -0.2792,   # ring
            -0.2792, -0.2792, -0.2792, -0.2792,   # thumb
        ], dtype=np.float32)

        q_max = np.array([
             0.5585,  0.5585,  0.5585,  1.5707,   # index
             1.7278,  1.7278,  1.7278,  1.1519,   # middle
             1.7278,  1.7278,  1.7278,  1.7278,   # ring
             1.7278,  1.7278,  1.7278,  1.7627,   # thumb
        ], dtype=np.float32)

        policy = RslRlPolicyWrapperRealWorld(
            model_path=self.policy_path,
            device=str(self.device),
            q_min=q_min,
            q_max=q_max,
            alpha=0.95,
            action_scale=self.get_parameter('action_scale').value
        )

        return policy



    # Subscriber
    def joint_callback(self, msg: JointState):
        self.last_joint_state = msg

    def pose_callback(self, msg: PoseStamped):
        self.last_cube_pose = msg
    
    def send_zero_command(self):
        """Send zero position command to all 16 Allegro joints."""
        if self.last_joint_state is None:
            self.get_logger().warn("No joint state received yet. Cannot zero hand.")
            return

        zero_msg = JointState()
        zero_msg.header.stamp = self.get_clock().now().to_msg()
        zero_msg.name = list(self.last_joint_state.name)
        zero_msg.position = [0.0] * len(self.last_joint_state.position)

        zero_msg.position[12] = 0.28 # same as simulation default pose, thumb_joint_0 = 0.28

        # Publish multiple times to ensure hardware receives it
        for _ in range(10):
            self.joint_pub.publish(zero_msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info("Sent zero command to Allegro hand.")

    # Control Loop (30 Hz)
    def control_loop(self):

        if self.last_joint_state is None or self.last_cube_pose is None:
            return

        # -- joint positions (sim order) and velocities
        joint_state = np.array(self.last_joint_state.position, dtype=np.float32)
        joint_state = real2sim_joints(joint_state)
        joint_vel = np.array(self.last_joint_state.velocity, dtype=np.float32)
        joint_vel = real2sim_joints(joint_vel) * 0.2  # matches sim scale=0.2

        # -- normalize joint positions to [-1, 1] (matches joint_pos_limit_normalized in sim)
        q_min = self.policy._q_min.cpu().numpy()
        q_max = self.policy._q_max.cpu().numpy()
        joint_pos_norm = (2.0 * joint_state - q_min - q_max) / (q_max - q_min)

        # -- object pose (world frame: +0.5 in z to match sim env origin at hand root)
        position = self.last_cube_pose.pose.position
        cube_position = np.array([position.x, position.y, position.z], dtype=np.float32)
        cube_position[2] += 0.5
        orientation = self.last_cube_pose.pose.orientation
        cube_quat = np.array([orientation.w, orientation.x, orientation.y, orientation.z], dtype=np.float32)

        # -- object velocities via finite difference (sim: lin_vel scale=1.0, ang_vel scale=0.2)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._prev_cube_pos is not None:
            dt = max(now - self._prev_cube_time, 1e-3)
            object_lin_vel = (cube_position - self._prev_cube_pos) / dt
            dq = quat_mul(cube_quat, quat_conjugate(self._prev_cube_quat))
            w = float(np.clip(abs(dq[0]), 0.0, 1.0))
            sin_half = np.sqrt(max(0.0, 1.0 - w * w))
            if sin_half > 1e-7:
                angle = 2.0 * np.arccos(w)
                axis = dq[1:] / sin_half
                if dq[0] < 0:
                    axis = -axis
                object_ang_vel = (axis * angle / dt * 0.2).astype(np.float32)
            else:
                object_ang_vel = np.zeros(3, dtype=np.float32)
        else:
            object_lin_vel = np.zeros(3, dtype=np.float32)
            object_ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_cube_pos = cube_position.copy()
        self._prev_cube_quat = cube_quat.copy()
        self._prev_cube_time = now


        # -- command / action terms
        goal_pose = np.concatenate([self.goal_pos, self.goal_quat]).copy()
        quat_difference, diff_angle = goal_quat_diff(cube_quat, self.goal_quat)
        last_action = self.last_raw_action

        self.get_logger().info(f"{diff_angle}")

        if diff_angle < self.orientation_success_threshold:
            self.get_logger().info("SUCCESS..SAMPLING NEW GOAL!")
            self.goal_pos, self.goal_quat = generate_goal_pose()
            goal_pose = np.concatenate([self.goal_pos, self.goal_quat]).copy()
            quat_difference, diff_angle = goal_quat_diff(cube_quat, self.goal_quat)

        # Concatenate in exactly the same order as KinematicObsGroupCfg (76 dims total):
        # joint_pos(16) joint_vel(16) object_pos(3) object_quat(4)
        # object_lin_vel(3) object_ang_vel(3) fingertip_dists(4)
        # goal_pose(7) goal_quat_diff(4) last_action(16)
        obs = np.concatenate([
            joint_pos_norm,    # 16
            joint_vel,         # 16
            cube_position,     # 3
            cube_quat,         # 4
            object_lin_vel,    # 3
            object_ang_vel,    # 3
            # fingertip_dists removed: task4 checkpoint trained without this term (72 dims)
            goal_pose,         # 7
            quat_difference,   # 4
            last_action,       # 16
        ])  # total: 72

        if not self.initialized_buffer:
            self.policy.reset()
            self.policy.set_prev_action(joint_state)
            self.initialized_buffer = True

        with torch.inference_mode():
            raw_action_tensor, q_cmd = self.policy(obs)

        self.last_raw_action = raw_action_tensor.reshape(16).cpu().numpy()


        # Publish command
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = list(self.last_joint_state.name)
        cmd_msg.position = q_cmd.tolist()

        self.joint_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
