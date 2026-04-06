# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def success_bonus(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal.

    The object is considered to have reached the goal when the object orientation is within the threshold.
    The reward is 1.0 if the object has reached the goal, otherwise 0.0.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # obtain the threshold for the orientation error
    threshold = command_term.cfg.orientation_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return dtheta <= threshold


def track_pos_l2(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for tracking the object position using the L2 norm.

    The reward is the distance between the object position and the goal position.

    Args:
        env: The environment object.
        command_term: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins

    return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object orientation using the inverse of the orientation error.

    The reward is the inverse of the orientation error between the object orientation and the goal orientation.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        rot_eps: The threshold for the orientation error. Default is 1e-3.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return 1.0 / (dtheta + rot_eps)

def object_linvel_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object linear velocity (L2 norm).

    Penalizes translational motion of the object to encourage stable grasping.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.norm(asset.data.root_lin_vel_w, dim=-1)


def object_angvel_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object angular velocity (L2 norm).

    Penalizes rotational motion of the object to discourage uncontrolled spinning.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)


def object_to_hand_dist(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for the distance between the object and the robot hand root.

    Encourages the hand to keep the object close and penalizes dropping.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    return torch.linalg.norm(asset.data.root_pos_w - robot.data.root_pos_w, dim=-1)


def object_height_below_threshold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """Penalty when the object falls below a height threshold.

    Returns 1.0 for each environment where the object z-position (in world frame)
    has dropped below ``threshold``, and 0.0 otherwise.

    Args:
        threshold: World-frame z height below which the object is considered dropped.
                   Default is 0.3 m (well below the hand root at z=0.5).
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return (asset.data.root_pos_w[:, 2] < threshold).float()



def object_spin_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for excessive object spin (always active).

    Returns the L2 norm of the object's angular velocity to discourage unnecessary
    spinning
    TODO: implement a penalty term for excessive object spin."""
    asset: RigidObject = env.scene[object_cfg.name]
    # angular velocity in world frame, shape: (num_envs, 3)
    return torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)


def object_spin_near_goal_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    angle_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalty for excessive object spin (active only when near the goal orientation)

    Returns the L2 norm of the object's angular velocity masked by whether the
    object is within angle_threshold radians of the goal. This settles the policy smoothly 
    once close to the target.
    TODO: implement a spin penalty that is only active near the goal orientation."""
    
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # compute orientation error between current and goal
    goal_quat_w = command_term.command[:, 3:7]
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    # binary mask: 1.0 when within threshold, 0.0 otherwise
    near_goal = (dtheta <= angle_threshold).float()

    # L2 norm of angular velocity, zeroed out when far from goal
    ang_vel_magnitude = torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)
    return near_goal * ang_vel_magnitude

def object_spin_xz_axis_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
):
    """Penalize the x and z axis direction twisting because they seem to be the hardest in the visualization"""
    """Penalize the angular velocity that the object obtains"""
    asset: RigidObject = env.scene[object_cfg.name]

    # (num_envs, 3) -> [wx, wy, wz]
    ang_vel = asset.data.root_ang_vel_w

    # keep only x and z components
    ang_vel_xz = ang_vel[:, [0, 2]]

    ## L2 Norm
    return torch.linalg.norm(ang_vel_xz, dim=-1)