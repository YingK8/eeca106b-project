#!/usr/bin/env python3
"""
Main script to run visual servoing
Author: Daniel Municio, Spring 2026
"""

import rclpy
from rclpy.node import Node
import numpy as np
from tf2_ros import TransformListener, Buffer
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
import argparse
import sys
import select
import time

from .trajectories import LinearTrajectory, CircularTrajectory
from .controller import UR7eTrajectoryController, PIDJointVelocityController


class VisualServo(Node):
    def __init__(self, args):
        super().__init__('visual_servo_node')

        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trajectory = None
        self.trajectory_start_time = None
        self.current_joint_state = None

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        # MoveIt IK client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.controller_type = args.controller
        if self.controller_type == 'default':
            self.trajectory_controller = UR7eTrajectoryController(self)
        elif self.controller_type == 'pid':

            # PID gains tuned for UR7e (Use these as a start)

            Kp = 0.3 * np.array([0.4, 2, 1.7, 1.5, 2, 2])
            Kd = 0.005 * np.array([2, 1, 2, 0.5, 0.8, 0.8])
            Ki = 0.02 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6])

            self.velocity_controller = PIDJointVelocityController(self, Kp, Ki, Kd)

        self.path_pub = self.create_publisher(Path, '/trajectory_path', 1)

        self.viz_timer = None
        self.ar_tag_detected = False
        self.ar_tag_position = None

        self.get_logger().info("Visual Servo Node initialized")
        self.get_logger().info(f"Task: {args.task}")
        self.get_logger().info(f"AR Marker ID: {args.ar_marker}")
        self.get_logger().info(f"Controller: {args.controller}")

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """
        Compute IK for a given workspace pose using MoveIt.

        Parameters
        ----------
        x, y, z : float
            Target position in base_link frame
        qx, qy, qz, qw : float
            Target orientation as quaternion (default: gripper pointing down)

        Returns
        -------
        JointState or None
            Joint state solution if IK succeeds, None otherwise
        """
        if self.current_joint_state is None:
            self.get_logger().error("No joint state available")
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'ur_manipulator'
        ik_req.ik_request.robot_state.joint_state = self.current_joint_state
        ik_req.ik_request.ik_link_name = 'wrist_3_link'
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.timeout = Duration(sec=2)
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        # Debug: print joint names from IK solution
        if not hasattr(self, '_printed_ik_joint_names'):
            self.get_logger().info(f'IK solution joint names: {result.solution.joint_state.name}')
            self.get_logger().info(f'Current joint_state names: {self.current_joint_state.name if self.current_joint_state else "None"}')
            self._printed_ik_joint_names = True

        return result.solution.joint_state

    def lookup_ar_tag(self, marker_id, timeout=5.0):
        target_frame = 'base_link'
        source_frame = f'ar_marker_{marker_id}'

        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < timeout:
            try:
                transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
                translation = transform.transform.translation
                position_wrist_frame = np.array([translation.x, translation.y, translation.z])
                return position_wrist_frame

            except Exception as e:
                self.get_logger().info(f"Waiting for AR tag {marker_id}... ({time.time() - start_time:.1f}s)")
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(f"Bonk: Could not find AR tag {marker_id} after {timeout}s")
        return None

    def create_trajectory(self):
        """
        Returns:

        Trajectory or None
            Trajectory object if successful, None otherwise
        """
        ar_position = self.lookup_ar_tag(self.args.ar_marker)

        if ar_position is None:
            self.get_logger().error(f"Bonk: Could not find AR tag {self.args.ar_marker}")
            return None

        hover_height = 0.25  # meters above AR tag
        goal_position = ar_position + np.array([0, 0, hover_height])

        if self.args.task == 'line':
            try:
                wrist_transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    'wrist_3_link',
                    rclpy.time.Time()
                )
                wrist_pos = wrist_transform.transform.translation
                start_position = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])
                self.get_logger().info(f"  Using current wrist position as start")
            except Exception as e:
                self.get_logger().warn(f"Could not get current wrist position: {e}")

            self.get_logger().info(f"  Task: LINEAR")
            self.get_logger().info(f"  Start position (base_link frame): {start_position}")
            self.get_logger().info(f"  Distance: {np.linalg.norm(goal_position - start_position):.3f} m")

            trajectory = LinearTrajectory(
                start_position=start_position,
                goal_position=goal_position,
                total_time=self.args.total_time
            )

        elif self.args.task == 'circle':
            # Circular trajectory around goal position
            self.get_logger().info(f"  Task: CIRCULAR")
            self.get_logger().info(f"  Center position (base_link frame): {goal_position}")
            self.get_logger().info(f"  Radius: {self.args.circle_radius} m")
            self.get_logger().info(f"  Circumference: {2 * np.pi * self.args.circle_radius:.3f} m")


            trajectory = CircularTrajectory(
                center_position=goal_position,
                radius=self.args.circle_radius,
                total_time=self.args.total_time
            )

        else:
            self.get_logger().error(f"Bonk task: {self.args.task}")
            return None

        return trajectory



    def publish_trajectory_visualization(self):
        """Publish trajectory as a Path for RViz visualization"""
        if self.trajectory is None:
            return

        num_vis_points = 100
        times = np.linspace(0, self.trajectory.total_time, num_vis_points)

        path_msg = Path()
        path_msg.header.frame_id = 'base_link'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for t in times:
            desired_pose = self.trajectory.target_pose(t)

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            pose_stamped.pose.position.x = desired_pose[0]
            pose_stamped.pose.position.y = desired_pose[1]
            pose_stamped.pose.position.z = desired_pose[2]

            pose_stamped.pose.orientation.x = desired_pose[3]
            pose_stamped.pose.orientation.y = desired_pose[4]
            pose_stamped.pose.orientation.z = desired_pose[5]
            pose_stamped.pose.orientation.w = desired_pose[6]

            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def visualization_callback(self):
        """Timer callback to publish all visualizations"""
        self.publish_trajectory_visualization()

    def start_visualization_timer(self):
        """Start publishing trajectory visualization on a timer"""
        if self.viz_timer is not None:
            return  # Already running

        self.viz_timer = self.create_timer(1.0, self.visualization_callback)
        self.get_logger().info("Started trajectory visualization timer (publishing at 1 Hz)")

    def execute_trajectory(self):
        """
        Execute the trajectory by computing waypoints and sending to robot.
        """
        if self.trajectory is None:
            self.get_logger().error("No trajectory to execute")
            return

        self.get_logger().info("Starting trajectory execution...")


        num_waypoints = 20
        times = np.linspace(0, self.trajectory.total_time, num_waypoints)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.get_logger().info(f"Computing IK for {num_waypoints} waypoints...")

        for i, t in enumerate(times):
            self.get_logger().info(f"  Computing waypoint {i+1}/{num_waypoints}...")

            desired_pose = self.trajectory.target_pose(t)

            x, y, z = desired_pose[0:3]
            qx, qy, qz, qw = desired_pose[3:7]

            joint_solution = self.compute_ik(x, y, z, qx, qy, qz, qw)

            if joint_solution is None:
                self.get_logger().error(f"IK failed at waypoint {i+1}/{num_waypoints}")
                return

            point = JointTrajectoryPoint()
            ik_joint_dict = {}

            for i, name in enumerate(joint_solution.name):
                if i < 6:
                    ik_joint_dict[name] = joint_solution.position[i]

            point.positions = [
                ik_joint_dict['shoulder_pan_joint'],
                ik_joint_dict['shoulder_lift_joint'],
                ik_joint_dict['elbow_joint'],
                ik_joint_dict['wrist_1_joint'],
                ik_joint_dict['wrist_2_joint'],
                ik_joint_dict['wrist_3_joint']
            ]

            point.velocities = [0.0] * 6

            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)

            joint_traj.points.append(point)

        # Compute velocities using finite differences
        for i in range(len(joint_traj.points)):
            if i == 0:
                # Forward difference for first point
                if len(joint_traj.points) > 1:
                    dt = (joint_traj.points[1].time_from_start.sec - joint_traj.points[0].time_from_start.sec) + \
                         (joint_traj.points[1].time_from_start.nanosec - joint_traj.points[0].time_from_start.nanosec) * 1e-9
                    if dt > 0:
                        dp = np.array(joint_traj.points[1].positions) - np.array(joint_traj.points[0].positions)
                        joint_traj.points[i].velocities = list(dp / dt)
            elif i == len(joint_traj.points) - 1:
                # Backward difference for last point
                dt = (joint_traj.points[i].time_from_start.sec - joint_traj.points[i-1].time_from_start.sec) + \
                     (joint_traj.points[i].time_from_start.nanosec - joint_traj.points[i-1].time_from_start.nanosec) * 1e-9
                if dt > 0:
                    dp = np.array(joint_traj.points[i].positions) - np.array(joint_traj.points[i-1].positions)
                    joint_traj.points[i].velocities = list(dp / dt)
            else:
                # Central difference for middle points
                dt = (joint_traj.points[i+1].time_from_start.sec - joint_traj.points[i-1].time_from_start.sec) + \
                     (joint_traj.points[i+1].time_from_start.nanosec - joint_traj.points[i-1].time_from_start.nanosec) * 1e-9
                if dt > 0:
                    dp = np.array(joint_traj.points[i+1].positions) - np.array(joint_traj.points[i-1].positions)
                    joint_traj.points[i].velocities = list(dp / dt)

        # Move to start position first to avoid tolerance violations
        self.get_logger().info("Moving to trajectory start position...")
        self._move_to_start(joint_traj.points[0].positions)

        self._execute_joint_trajectory(joint_traj)

    def _move_to_start(self, start_positions):
        """
        Move robot to the start position of the trajectory.
        This ensures the robot is already at the first waypoint before executing the full trajectory.

        Parameters
        ----------
        start_positions : list
            Joint positions for the start configuration
        """
        # For line task, skip move to start (assumes robot starts at current position)
        if self.args.task == 'line':
            self.get_logger().info('Line task: skipping move to start, using current position')
            return

        joint_traj = JointTrajectory()
        joint_traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = list(start_positions)
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 3  # Take 3 seconds to reach start
        point.time_from_start.nanosec = 0

        joint_traj.points.append(point)

        # Execute the move to start
        success = self.trajectory_controller.execute_joint_trajectory(joint_traj)
        if success:
            self.get_logger().info('Reached trajectory start position')
        else:
            self.get_logger().error('Failed to reach trajectory start position')

    def _execute_joint_trajectory(self, joint_traj):
        """Execute trajectory using the selected controller"""
        if self.controller_type == 'default':
            success = self.trajectory_controller.execute_joint_trajectory(joint_traj)
            if not success:
                self.get_logger().error('Trajectory execution failed')
        else:
            self._execute_velocity_control(joint_traj)

    def _execute_velocity_control(self, joint_traj):
        """
        Execute trajectory using velocity control 
        Uses a ROS2 timer for proper 100Hz control.

        Parameters
        ----------
        joint_traj : JointTrajectory
            The trajectory to follow
        """

        # Store trajectory for timer callback
        self._control_joint_traj = joint_traj
        self._control_current_index = 0
        self._control_max_index = len(joint_traj.points) - 1
        self._control_iteration = 0
        self._control_start_time = self.get_clock().now()
        self._control_done = False

        # Create velocity command publisher
        self._velocity_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        self.get_logger().info(f'Executing trajectory with {self.velocity_controller.get_name()} controller...')
        self.get_logger().info(f'Trajectory has {len(joint_traj.points)} waypoints')

        # Print first few waypoint velocities
        for i in range(min(3, len(joint_traj.points))):
            self.get_logger().info(f'Waypoint {i}: vel = {joint_traj.points[i].velocities}')

        # Create timer at 10 Hz
        self._control_timer = self.create_timer(0.1, self._velocity_control_callback)

        # Wait for trajectory to complete
        while rclpy.ok() and not self._control_done:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop the robot
        vel_msg = Float64MultiArray()
        vel_msg.data = [0.0] * 6
        self._velocity_pub.publish(vel_msg)

        # Clean up timer
        self._control_timer.cancel()

        self.get_logger().info('Velocity control execution complete!')

    def _velocity_control_callback(self):
        """Timer callback for velocity control at 10 Hz"""
        # Find elapsed time
        elapsed = (self.get_clock().now() - self._control_start_time).nanoseconds / 1e9

        # Get current joint state
        if self.current_joint_state is None:
            self.get_logger().warn('No joint state available, waiting...')
            return

        # Reorder current joint state to match trajectory order
        current_joint_dict = {}
        for i, name in enumerate(self.current_joint_state.name):
            if i < 6:
                current_joint_dict[name] = (self.current_joint_state.position[i],
                                            self.current_joint_state.velocity[i])

        current_position = np.array([
            current_joint_dict['shoulder_pan_joint'][0],
            current_joint_dict['shoulder_lift_joint'][0],
            current_joint_dict['elbow_joint'][0],
            current_joint_dict['wrist_1_joint'][0],
            current_joint_dict['wrist_2_joint'][0],
            current_joint_dict['wrist_3_joint'][0]
        ])
        current_velocity = np.array([
            current_joint_dict['shoulder_pan_joint'][1],
            current_joint_dict['shoulder_lift_joint'][1],
            current_joint_dict['elbow_joint'][1],
            current_joint_dict['wrist_1_joint'][1],
            current_joint_dict['wrist_2_joint'][1],
            current_joint_dict['wrist_3_joint'][1]
        ])

        # Interpolate trajectory to get target at current time
        target_position, target_velocity, self._control_current_index = self._interpolate_trajectory(
            self._control_joint_traj, elapsed, self._control_current_index
        )

        # Compute control command
        commanded_velocity = self.velocity_controller.step_control(
            target_position, target_velocity, current_position, current_velocity
        )


        vel_msg = Float64MultiArray()
        vel_msg.data = commanded_velocity.tolist()
        self._velocity_pub.publish(vel_msg)

        if self._control_current_index >= self._control_max_index:
            self.get_logger().info(f'Reached max index at t={elapsed:.2f}s')
            self._control_done = True

        self._control_iteration += 1

    def _interpolate_trajectory(self, joint_traj, t, current_index=0):
        """
        Interpolate trajectory waypoints based on current time.
        Based on interpolate_path from 106a Lab 7 Fall 2024.

        Parameters
        ----------
        joint_traj : JointTrajectory
            The trajectory with waypoints
        t : float
            Current time in seconds
        current_index : int
            Waypoint index from which to start search

        Returns
        -------
        tuple
            (target_position, target_velocity, current_index)
        """
        epsilon = 0.0001
        max_index = len(joint_traj.points) - 1

        # If time at current index is greater than current time, start from beginning
        current_time = joint_traj.points[current_index].time_from_start.sec + \
                      joint_traj.points[current_index].time_from_start.nanosec * 1e-9
        if current_time > t:
            current_index = 0

        # Iterate forward to find the right time bracket
        while (current_index < max_index and
               joint_traj.points[current_index + 1].time_from_start.sec +
               joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9 < t + epsilon):
            current_index += 1

        # Perform linear interpolation
        if current_index < max_index:
            time_low = joint_traj.points[current_index].time_from_start.sec + \
                      joint_traj.points[current_index].time_from_start.nanosec * 1e-9
            time_high = joint_traj.points[current_index + 1].time_from_start.sec + \
                       joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9

            target_position_low = np.array(joint_traj.points[current_index].positions)
            target_velocity_low = np.array(joint_traj.points[current_index].velocities)

            target_position_high = np.array(joint_traj.points[current_index + 1].positions)
            target_velocity_high = np.array(joint_traj.points[current_index + 1].velocities)

            # Linear interpolation
            alpha = (t - time_low) / (time_high - time_low) if time_high != time_low else 0
            target_position = target_position_low + alpha * (target_position_high - target_position_low)
            target_velocity = target_velocity_low + alpha * (target_velocity_high - target_velocity_low)
        else:
            # At last waypoint
            target_position = np.array(joint_traj.points[current_index].positions)
            target_velocity = np.array(joint_traj.points[current_index].velocities)

        return target_position, target_velocity, current_index

    def run(self):
        self.trajectory = self.create_trajectory()

        if self.trajectory is None:
            self.get_logger().error("Failed to create trajectory")
            return

        self.get_logger().info("Trajectory created! Publishing visualization to RViz...")
        self.start_visualization_timer()

        self.get_logger().info("\nPress ENTER to execute trajectory (Ctrl+C to cancel)")
        while rclpy.ok():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.execute_trajectory()
        self.get_logger().info("=== Execution Complete ===")


def main(args=None):
    parser = argparse.ArgumentParser(description='Bonk')
    parser.add_argument('--task', '-t', type=str, default='line',
                       choices=['line', 'circle'],
                       help='Type of trajectory: line or circle')
    parser.add_argument('--ar_marker', type=int, default=0,
                       help='AR marker ID to track')
    parser.add_argument('--total_time', type=float, default=10.0,
                       help='Total time for trajectory execution (seconds)')
    parser.add_argument('--circle_radius', type=float, default=0.1,
                       help='Radius for circular trajectory (meters)')
    parser.add_argument('--controller', '-c', type=str, default='default',
                       choices=['default', 'pid'],
                       help='Controller type: trajectory (default), pid')

    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = VisualServo(parsed_args)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
