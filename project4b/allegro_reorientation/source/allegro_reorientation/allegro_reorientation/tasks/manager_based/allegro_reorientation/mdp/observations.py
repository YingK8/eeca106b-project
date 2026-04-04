# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat

def fingertip_to_object_distances(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """TODO: implement fingertip-to-object distances for the selected fingertip bodies."""
    
    finger_tips = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
    
    robot_cfg.resolve(env.scene)
    object_cfg.resolve(env.scene)
    
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    all_body_names = robot.data.body_names
    finger_indices = [all_body_names.index(name) for name in finger_tips]

    tip_poses_w = robot.data.body_pos_w[:, finger_indices, :]
    obj_pos_w = obj.data.root_pos_w

    distances = torch.linalg.norm(tip_poses_w - obj_pos_w[:, None, :], dim=-1)
    return distances
