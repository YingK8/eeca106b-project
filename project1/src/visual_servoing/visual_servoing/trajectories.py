#!/usr/bin/env python3
"""
Trajectory classes for Project 1 Part 1
Defines Linear and Circular trajectories for the UR7e end effector

Author: EECS 106B Course Staff, Spring 2026
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse


class Trajectory:
    """Base trajectory class"""

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            Duration of the trajectory in seconds
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """
        raise NotImplementedError

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """
        raise NotImplementedError

    def display_trajectory(self, num_waypoints=100, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animation : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0],
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0],
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0],
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0],
                        target_positions[:, 1],
                        target_positions[:, 2],
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0],
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0],
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0],
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0],
                        target_velocities[:, 1],
                        target_velocities[:, 2],
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints,
                                                          fargs=([line0, line1],),
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))),
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)


class LinearTrajectory(Trajectory):
    """
    Straight line trajectory from start to goal position.
    Uses trapezoidal velocity profile (constant acceleration, constant velocity, constant deceleration).
    """

    def __init__(self, start_position, goal_position, total_time):
        """
        Parameters
        ----------
        start_position : np.ndarray
            3D starting position [x, y, z] in meters
        goal_position : np.ndarray
            3D goal position [x, y, z] in meters
        total_time : float
            Total duration of trajectory in seconds
        """
        super().__init__(total_time)

        self.start_position = np.array(start_position)
        self.goal_position = np.array(goal_position)
        self.distance = self.goal_position - self.start_position
        self.acceleration = (self.distance * 4.0) / (self.total_time ** 2)  # keep constant magnitude acceleration
        self.v_max = (self.total_time / 2.0) * self.acceleration  # maximum velocity magnitude
        self.desired_orientation = np.array([0, 1, 0, 0])
        self.t_half = self.total_time / 2.0


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are
        the desired end-effector position, and the last four entries are the
        desired end-effector orientation as a quaternion, all written in the
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds
        to the quaternion [0, 1, 0, 0].

        Parameters
        ----------
        time : float

        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate the position of the end effector at time t,
            # For the first half of the trajectory, maintain a constant acceleration
            pos = (
                self.start_position
                + 0.5 * self.acceleration * time**2 
            )
        else:
            # TODO: Calculate the position of the end effector at time t,
            # For the second half of the trajectory, maintain a constant acceleration
            # Hint: Calculate the remaining distance to the goal position.
            pos_mid = (
                self.start_position
                + 0.5 * self.acceleration * self.t_half**2 
            )
            t_remain = time - self.t_half

            d_pos = (
                self.v_max * t_remain
                - 0.5 * self.acceleration * t_remain**2
            )

            pos = pos_mid + d_pos

        return np.hstack((pos, self.desired_orientation))

    def target_velocity(self, time):
        """
        Returns the end effector's desired velocity at time t as a 6D twist
        [vx, vy, vz, wx, wy, wz] in the world frame.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired velocity of the end effector (linear + angular)
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate velocity using the acceleration and time
            # For the first half of the trajectory, we maintain a constant acceleration

            linear_vel = self.acceleration * time
        else:
            # TODO: start slowing the velocity down from the maximum one
            # For the second half of the trajectory, maintain a constant deceleration
            linear_vel = (
                self.v_max
                - self.acceleration * (time - self.t_half)
            ) 
        return np.hstack((linear_vel, np.zeros(3)))


class CircularTrajectory(Trajectory):
    """
    Circular trajectory around a center point in a horizontal plane.
    Uses angular trapezoidal velocity profile.
    """

    def __init__(self, center_position, radius, total_time):
        """
        Parameters
        ----------
        center_position : np.ndarray
            3D center position [x, y, z] in meters
        radius : float
            Radius of circle in meters
        total_time : float
            Total duration of trajectory in seconds
        """
        super().__init__(total_time)

        self.center_position = np.array(center_position)
        self.radius = radius
        self.angular_acceleration = (2 * np.pi * 4.0) / (self.total_time ** 2)  # keep constant magnitude acceleration
        self.angular_v_max = (self.total_time / 2.0) * self.angular_acceleration  # maximum velocity magnitude
        self.desired_orientation = np.array([0, 1, 0, 0])
        self.t_half = self.total_time / 2.0

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are
        the desired end-effector position, and the last four entries are the
        desired end-effector orientation as a quaternion, all written in the
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds
        to the quaternion [0, 1, 0, 0].

        Parameters
        ----------
        time : float

        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate the ANGLE of the end effector at time t,
            # For the first half of the trajectory, maintain a constant acceleration


            theta = 0.5 * self.angular_acceleration * time**2
        else:
            # TODO: Calculate the ANGLE of the end effector at time t,
            # For the second half of the trajectory, maintain a constant acceleration
            # Hint: Calculate the remaining angle to the goal position.


            theta_mid = (
                
                0.5 * self.angular_acceleration * self.t_half**2 
            )

            t_remain = time - self.t_half

            d_theta = (
                self.angular_v_max * t_remain
                - 0.5 * self.angular_acceleration * t_remain**2
            )

            theta = theta_mid + d_theta


        
        pos_d = np.ndarray.flatten(self.center_position + self.radius * np.array([np.cos(theta), np.sin(theta), 0]))
        return np.hstack((pos_d, self.desired_orientation))

    def target_velocity(self, time):
        """
        Returns the end effector's desired velocity at time t as a 6D twist
        [vx, vy, vz, wx, wy, wz] in the world frame.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired velocity of the end effector (linear + angular)
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate ANGULAR position and velocity using the acceleration and time
            # For the first half of the trajectory, we maintain a constant acceleration


            theta = 0.5 * self.angular_acceleration*time**2
            theta_dot = self.angular_acceleration*time
        else:
            # TODO: start slowing the ANGULAR velocity down from the maximum one
            # For the second half of the trajectory, maintain a constant deceleration

            theta_mid = (
                
                0.5 * self.angular_acceleration * self.t_half**2 
            )

            t_remain = time - self.t_half

            d_theta = (
                self.angular_v_max * t_remain
                - 0.5 * self.angular_acceleration * t_remain**2
            )

            theta = theta_mid + d_theta

            theta_dot_mid = self.angular_acceleration*self.t_half
            theta_dot = theta_dot_mid - self.angular_acceleration * (time - self.t_half)
            
        vel_d = np.ndarray.flatten(self.radius * theta_dot * np.array([-np.sin(theta), np.cos(theta), 0]))
        return np.hstack((vel_d, np.zeros(3)))


def define_trajectories(args):
    """Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        # Example linear trajectory
        start = np.array([0.3, 0.2, 0.3])
        goal = np.array([0.5, 0.4, 0.4])
        total_time = 5.0
        trajectory = LinearTrajectory(start, goal, total_time)
    elif args.task == 'circle':
        # Example circular trajectory
        center = np.array([0.4, 0.3, 0.3])
        radius = 0.1
        total_time = 8.0
        trajectory = CircularTrajectory(center, radius, total_time)
    return trajectory


if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the
    animate function to visualize the full trajectory in a 3D plot.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)

    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)
