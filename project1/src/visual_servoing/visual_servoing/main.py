#!/usr/bin/env python3
"""
Main script to run visual servoing
Author: Daniel Municio, Spring 2026
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import subprocess
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
import subprocess
try:
    import matplotlib
    # requires python3-tk for Ubuntu
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

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
        self.is_callback_running = False

        self.hover_height = 0.5

        # Use ReentrantCallbackGroup to prevent service call deadlocks
        self.cb_group = ReentrantCallbackGroup()

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        # MoveIt IK client - put in Reentrant group so it doesn't block
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.cb_group)
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.controller_type = args.controller
        # Always create trajectory controller for moving to start position
        self.trajectory_controller = UR7eTrajectoryController(self)

        if self.controller_type == 'pid':
            # PID gains tuned for UR7e (Use these as a start)

            # Change Kp's shoulder pan from 0.4 -> 1.5
            Kp = 2 * np.array([1.5, 2, 1.7, 1.5, 2, 2])
            # Change Kd's shoulder_pan from 2 -> 1.5
            Kd = 0.1 * np.array([1.5, 1, 2, 0.5, 0.8, 0.8])
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
        self.get_logger().info(f"Log: {args.log}")

        # Logging variables
        self.log_enabled = args.log
        if self.log_enabled:
            self.log_times = []
            self.log_actual_positions = []
            self.log_actual_velocities = []
            self.log_target_positions = []
            self.log_target_velocities = []
            
            # Initialize log storage
            self.log_data = {
                'time': [],
                'error': [],
                'joint_velocities': []
            }
            self.start_time = time.time()

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
        
        # This function spins the node. If called from a callback, it causes deadlock.
        # Ensure this is only called from the main thread loop.
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

            except Exception:
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

        hover_height = 0.75  # meters above AR tag
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

    def compute_vertical_offset_target(self, marker_id, vertical_offset=0.3):
        """
        Compute target end-effector pose to maintain vertical offset above AR marker.

        Returns
        -------
        tuple or None
            (target_position, target_orientation) where:
            - target_position: np.ndarray (3,) - desired wrist position in base_link frame
            - target_orientation: np.ndarray (4,) - desired quaternion [qx, qy, qz, qw]
            Returns None if AR marker lookup fails
        """
        try:
            target_frame = 'base_link'
            source_frame = f'ar_marker_{marker_id}'

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            translation = transform.transform.translation
            ar_position = np.array([translation.x, translation.y, translation.z])

            # Update most recent valid position
            self._vs_last_valid_target = ar_position

        except Exception:
            if self._vs_last_valid_target is None:
                self.get_logger().warn('AR marker lost and no previous position available')
                return None

            ar_position = self._vs_last_valid_target
            self.get_logger().warn(f'AR marker not visible, using last known position: {ar_position}')

        # target = marker + offset
        target_position = ar_position + np.array([0, 0, vertical_offset])

        # Fixed orientation: gripper pointing down
        target_orientation = np.array([0.0, 1.0, 0.0, 0.0])

        return target_position, target_orientation

    def _switch_to_scaled_controller(self):
        """Activate the scaled_joint_trajectory_controller (default trajectory controller)."""
        try:
            cmd = [
                'ros2', 'control', 'switch_controllers',
                '--deactivate', 'forward_velocity_controller',
                '--activate', 'scaled_joint_trajectory_controller'
            ]
            subprocess.run(cmd, check=True)
            self.get_logger().info('Switched to scaled_joint_trajectory_controller')
        except Exception as e:
            self.get_logger().warn(f'Failed to switch to scaled controller: {e}')

    def _switch_to_forward_controller(self):
        """Activate the forward_velocity_controller (used by PID velocity controller)."""
        try:
            cmd = [
                'ros2', 'control', 'switch_controllers',
                '--activate', 'forward_velocity_controller',
                '--deactivate', 'scaled_joint_trajectory_controller'
            ]
            subprocess.run(cmd, check=True)
            self.get_logger().info('Switched to forward_velocity_controller')
        except Exception as e:
            self.get_logger().warn(f'Failed to switch to forward controller: {e}')

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
        self.is_callback_running = True


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

        try:
            for i, t in enumerate(times):
                self.get_logger().info(f"  Computing waypoint {i+1}/{num_waypoints}...")

                desired_pose = self.trajectory.target_pose(t)

                x, y, z = desired_pose[0:3]
                qx, qy, qz, qw = desired_pose[3:7]

                joint_solution = self.compute_ik(x, y, z, qx, qy, qz, qw)

                if joint_solution is None:
                    self.get_logger().error(f"IK failed at waypoint {i+1}/{num_waypoints}")
                    self.is_callback_running = False # Reset flag
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

        finally:
            # Always ensure we reset this flag
            self.is_callback_running = False

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
        # If drawing a circle with PID control, ensure the robot uses
        # the default scaled_joint_trajectory_controller to move to the start
        if self.args.task == 'circle' and self.controller_type == 'pid':
            self.get_logger().info('Activating scaled (default) controller before move to start')
            self._switch_to_scaled_controller()
        self._move_to_start(joint_traj.points[0].positions)

        # Switch to the appropriate ROS2 controller after moving to start
        switch_controllers(self.controller_type)

        self._execute_joint_trajectory(joint_traj)
        self._switch_to_scaled_controller()

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

        # Reset logs
        if self.log_enabled:
            self.log_times = []
            self.log_actual_positions = []
            self.log_actual_velocities = []
            self.log_target_positions = []
            self.log_target_velocities = []

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

        if self.log_enabled:
            self.plot_results(
                self.log_times,
                self.log_actual_positions,
                self.log_actual_velocities,
                self.log_target_positions,
                self.log_target_velocities
            )

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

        # Log data if enabled
        if self.log_enabled:
            self.log_times.append(elapsed)
            self.log_actual_positions.append(current_position)
            self.log_actual_velocities.append(current_velocity)
            self.log_target_positions.append(target_position)
            self.log_target_velocities.append(target_velocity)

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

    def _visual_servo_callback(self):
        """
        Visual servoing control step.
        Called directly from main loop (not a timer!) to avoid recursive spin.
        """

        if self.is_callback_running:
            return

        elapsed = (self.get_clock().now() - self._vs_start_time).nanoseconds / 1e9
        print("CALLBACK running at t =", elapsed)

        # current joint state
        if self.current_joint_state is None:
            self.get_logger().warn('No joint state available, waiting...')
            return

        # Reorder current joint state to match standard order
        current_joint_dict = {}
        for i, name in enumerate(self.current_joint_state.name):
            if i < 6:
                current_joint_dict[name] = (
                    self.current_joint_state.position[i],
                    self.current_joint_state.velocity[i]
                )

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
        
        # Compute target pose from AR marker position
        target_result = self.compute_vertical_offset_target(
            self.args.ar_marker,
            vertical_offset=self.hover_height
        )

        if target_result is None:
            self.get_logger().error('Failed to compute target pose, stopping')
            self._vs_done = True
            return

        target_position_ee, target_orientation_ee = target_result

        # Get transform of 'tool0' relative to 'base_link'
        tool0_transform = self.tf_buffer.lookup_transform(
            'base_link',       # Target frame (fixed frame)
            'tool0',           # Source frame (the end effector)
            rclpy.time.Time()  # Get the latest available transform
        )
        
        # Access position
        t = tool0_transform.transform.translation
        position = np.array([t.x, t.y, t.z])


        #========================================================================================
        # mini steps:
        print("Target pose (position + orientation):", target_result)

        target_position_ee, target_orientation_ee = target_result

        # Compute IK for target end-effector pose
        x, y, z = target_position_ee
        qx, qy, qz, qw = target_orientation_ee


        e_vec = target_position_ee - position[0:3]
        e_norm = np.linalg.norm(e_vec)
        print("Distance is ", e_norm)
        if (e_norm < 0.01):
            #self._vs_done = True
            velocity = np.zeros(6)
            commanded_velocity = self.velocity_controller.step_control(
                current_position,
                velocity,
                current_position,
                current_velocity
            )

            # zero_vel_msg = Float64MultiArray()
            # zero_vel_msg.data = [0.0] * 6  # Create a list of 6 zeros
            # self._velocity_pub.publish(zero_vel_msg)
            vel_msg = Float64MultiArray()
            vel_msg.data = commanded_velocity.tolist()
            self._velocity_pub.publish(vel_msg)
            self.plot_results(
                self.log_times,
                self.log_actual_positions,
                self.log_actual_velocities,
                self.log_target_positions,
                self.log_target_velocities
            )
            return
        e_normalised = e_vec/ e_norm 
        goal_pos_scaled = position[0:3] + 0.1 * e_normalised
        
        # log the points:
        
        # Find elapsed time
        elapsed = (self.get_clock().now() - self._control_start_time).nanoseconds / 1e9
        
        if self.log_enabled:
            target_position, target_velocity, self._control_current_index = self._interpolate_trajectory(
                self._control_joint_traj, elapsed, self._control_current_index
            )
            self.log_times.append(elapsed)
            self.log_actual_positions.append(position[0:3])
            self.log_actual_velocities.append(current_velocity)
            self.log_target_positions.append(target_position) # or maybe goal_pos_scaled?
            self.log_target_velocities.append(target_velocity) # or maybe np.zeros(6) since we are doing position control? 
        #========================================================================================


        # Compute IK for target end-effector pose
        x, y, z = goal_pos_scaled
        qx, qy, qz, qw = target_orientation_ee

        # This call uses spin_until_future_complete.
        # It IS safe now because we are calling this function from the main loop,
        # not from inside a timer callback.
        joint_solution = self.compute_ik(x, y, z, qx, qy, qz, qw)

        print("IK joint solution:", joint_solution)

        if joint_solution is None:
            self.get_logger().warn('IK failed for current target, maintaining previous command')
            return

        # Extract target joint positions from IK solution
        ik_joint_dict = {}
        for i, name in enumerate(joint_solution.name):
            if i < 6:
                ik_joint_dict[name] = joint_solution.position[i]

        target_joint_position = np.array([
            ik_joint_dict['shoulder_pan_joint'],
            ik_joint_dict['shoulder_lift_joint'],
            ik_joint_dict['elbow_joint'],
            ik_joint_dict['wrist_1_joint'],
            ik_joint_dict['wrist_2_joint'],
            ik_joint_dict['wrist_3_joint']
        ])

        # Target velocity is zero (pure position control)
        target_joint_velocity = np.zeros(6)

        # Compute control command using PID controller
        commanded_velocity = self.velocity_controller.step_control(
            target_joint_position,
            target_joint_velocity,
            current_position,
            current_velocity
        )

        # Publish velocity command
        vel_msg = Float64MultiArray()
        vel_msg.data = commanded_velocity.tolist()
        self._velocity_pub.publish(vel_msg)
        print("Published velocity command:", commanded_velocity)

    def _interpolate_trajectory(self, joint_traj, t, current_index=0):
        """
        Interpolate trajectory waypoints based on current time.
        Based on interget_loggerpolate_path from 106a Lab 7 Fall 2024.

        Parameters
        ----------
        joint_traj : JointTrajectory
            The trajectory with waypoints
        t : float
            Current time in seconds
        current_index : int
            Waypoint index from which to start search

        Returns
        -------target_position_ee
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

    def plot_results(
        self,
        times,
        actual_positions,
        actual_velocities,
        target_positions,
        target_velocities
    ):
        """
        Plots results.
        """
        if plt is None:
            self.get_logger().error("Matplotlib not found, skipping plot.")
            return

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)

        joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow',
            'wrist_1', 'wrist_2', 'wrist_3'
        ]

        # Plot joint space
        plt.figure(figsize=(12, 16))
        joint_num = len(joint_names)
        for i, joint_name in enumerate(joint_names):
            # Position
            plt.subplot(joint_num, 2, 2*i + 1)
            plt.plot(times, actual_positions[:, i], label='Actual')
            plt.plot(times, target_positions[:, i], label='Desired', linestyle='--')
            plt.ylabel(f"{joint_name} Pos (rad)")
            if i == 0:
                plt.title("Joint Position Tracking")
            if i == joint_num - 1:
                plt.xlabel("Time (s)")
            plt.legend(loc='upper right', fontsize='small')

            # Velocity
            plt.subplot(joint_num, 2, 2*i + 2)
            plt.plot(times, actual_velocities[:, i], label='Actual')
            plt.plot(times, target_velocities[:, i], label='Desired', linestyle='--')
            plt.ylabel(f"{joint_name} Vel (rad/s)")
            if i == 0:
                plt.title("Joint Velocity Tracking")
            if i == joint_num - 1:
                plt.xlabel("Time (s)")
            plt.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        print("Close the plot window to continue")
        plt.show()

    def run(self):
        # Check if visual servoing task
        if self.args.task == 'visual_servo':
            self.get_logger().info("=== Starting Visual Servoing Mode ===")
            self.get_logger().info(f"Tracking AR marker {self.args.ar_marker}")
            self.get_logger().info(f"Maintaining vertical offset: 0.15 meters above marker")
            self.get_logger().info(f"Controller: {self.controller_type}")

            print("=== Starting Visual Servoing Mode ===")
            print(f"Tracking AR marker {self.args.ar_marker}")
            print(f"Maintaining vertical offset: 0.15 meters above marker")
            print(f"Controller: {self.controller_type}")

            if self.controller_type != 'pid':
                self.get_logger().error("Visual servoing requires --controller pid")
                print("Visual servoing requires --controller pid")
                return

            self.execute_visual_servoing()
            self.get_logger().info("=== Visual Servoing Complete ===")
            print("=== Visual Servoing Complete ===")
            return

        self.trajectory = self.create_trajectory()

        if self.trajectory is None:
            self.get_logger().error("Failed to create trajectory")
            print("Failed to create Trajectory ")
            return

        self.get_logger().info("Trajectory created! Publishing visualization to RViz...")
        print("Trajectory created! Publishing visualization to RViz...")
        self.start_visualization_timer()

        self.get_logger().info("\nPress ENTER to execute trajectory (Ctrl+C to cancel)")
        print("\nPress ENTER to execute trajectory (Ctrl+C to cancel)")
        while rclpy.ok():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.execute_trajectory()
        self.get_logger().info("=== Execution Complete ===")
        print("=== Execution Complete ===")

    def execute_visual_servoing(self):
        """
        Execute real-time visual servoing to track AR marker with vertical offset.
        
        Refactored to run as a main loop instead of a timer to avoid 
        recursive spinning in compute_ik.
        """
        # Switch to forward velocity controller
        self._switch_to_forward_controller()

        self._vs_mode = True  # Flag to indicate visual servoing mode
        self._vs_last_valid_target = None  # Last known valid AR marker position
        self._vs_done = False  # Flag to stop control loop
        self._vs_start_time = self.get_clock().now()

        # Reset PID controller integral term
        self.velocity_controller.integral_err = np.zeros(6)
        self.velocity_controller.prev_time = self.get_clock().now()

        # Reset logs
        if self.log_enabled:
            self.log_times = []
            self.log_actual_positions = []
            self.log_actual_velocities = []
            self.log_target_positions = []
            self.log_target_velocities = []

        self._velocity_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )

        self.get_logger().info('Looking for AR marker to start visual servoing...')
        ar_position = self.lookup_ar_tag(self.args.ar_marker, timeout=10.0)

        if ar_position is None:
            self.get_logger().error('Cannot start visual servoing: AR marker not found')
            self._switch_to_scaled_controller()
            return

        self._vs_last_valid_target = ar_position
        self.get_logger().info(f'AR marker found at {ar_position}')
        self.get_logger().info('Starting visual servoing control loop at 10 Hz...')
        self.get_logger().info('Press Ctrl+C to stop')

        # Initialize start time here (after blocking lookup) so elapsed time starts at 0
        self._vs_start_time = self.get_clock().now()

        # NOTE: Replaced timer with direct loop call
        try:
            while rclpy.ok() and not self._vs_done:
                # Update joint states and transforms
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # Execute logic
                self._visual_servo_callback()
                
                # Sleep to maintain roughly 10Hz
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            self.get_logger().info('Visual servoing interrupted by user')

        vel_msg = Float64MultiArray()
        vel_msg.data = [0.0] * 6
        self._velocity_pub.publish(vel_msg)

        self._switch_to_scaled_controller()

        self.get_logger().info('Visual servoing complete!')

        if self.log_enabled:
            self.plot_results(
                self.log_times,
                self.log_actual_positions,
                self.log_actual_velocities,
                self.log_target_positions,
                self.log_target_velocities
            )


def switch_controllers(controller_type):
    """
    Switch ROS2 controllers based on the selected controller type.

    Parameters
    ----------
    controller_type : str
        Either 'default' or 'pid'
    """
    if controller_type == 'default':
        cmd = [
            'ros2', 'control', 'switch_controllers',
            '--deactivate', 'forward_velocity_controller',
            '--activate', 'scaled_joint_trajectory_controller'
        ]
    else:  # pid
        cmd = [
            'ros2', 'control', 'switch_controllers',
            '--deactivate', 'scaled_joint_trajectory_controller',
            '--activate', 'forward_velocity_controller'
        ]

    print(f"Switching controllers for '{controller_type}' mode...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Switching controllers successful")
        else:
            print(f"Controller switch warning: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Controller switch timed out")
    except Exception as e:
        print(f"Controller switch failed: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(description='Bonk')
    parser.add_argument('--task', '-t', type=str, default='line',
                       choices=['line', 'circle', 'visual_servo'],
                       help='Type of trajectory: line, circle, or visual_servo')
    parser.add_argument('--ar_marker', type=int, default=0,
                       help='AR marker ID to track')
    parser.add_argument('--total_time', type=float, default=10.0,
                       help='Total time for trajectory execution (seconds)')
    parser.add_argument('--circle_radius', type=float, default=0.1,
                       help='Radius for circular trajectory (meters)')
    parser.add_argument('--controller', '-c', type=str, default='default',
                       choices=['default', 'pid'],
                       help='Controller type: trajectory (default), pid')
    parser.add_argument('--log', action='store_true', help='plots controller performance')

    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = VisualServo(parsed_args)
   
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        # Switch back to default controller on cleanup
        print("Cleaning up: switching back to default controller...")
        switch_controllers('default')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()