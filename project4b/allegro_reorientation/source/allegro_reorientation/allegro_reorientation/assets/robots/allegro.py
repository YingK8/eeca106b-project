# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro hand presets for the external task package.

This module keeps the USD path and the environment defaults that are coupled to
that USD together, so switching hand assets from the CLI is predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_THIS_DIR = Path(__file__).resolve().parent
_CUSTOM_USD_PATH = _THIS_DIR.parent / "usd" / "gazebo_allegro_hand_right_abs_noworld.usd"
_ISAACLAB_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd"
DEFAULT_ALLEGRO_HAND_PRESET = "isaaclab"
ALLEGRO_HAND_PRESET_CHOICES = ("custom", "isaaclab")


@dataclass(frozen=True)
class AllegroHandPreset:
    robot_cfg: ArticulationCfg
    clone_in_fabric: bool
    object_init_pos: tuple[float, float, float]
    action_scale: float
    default_joint_pos_fallback: dict[str, float]


def _make_allegro_hand_cfg(
    usd_path: str,
    root_rot: tuple[float, float, float, float],
    init_joint_pos: dict[str, float],
) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=root_rot,
            joint_pos=init_joint_pos,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


ALLEGRO_HAND_PRESETS = {
    "custom": AllegroHandPreset(
        robot_cfg=_make_allegro_hand_cfg(
            usd_path=str(_CUSTOM_USD_PATH),
            root_rot=(0.57358, 0.0, -0.81915, 0.0),
            init_joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
        ),
        clone_in_fabric=False,
        object_init_pos=(0.0, 0.0, 0.56),
        action_scale=1.0 / 6.0,
        default_joint_pos_fallback={"thumb_joint_0": 0.28},
    ),
    "isaaclab": AllegroHandPreset(
        robot_cfg=_make_allegro_hand_cfg(
            usd_path=str(_ISAACLAB_USD_PATH),
            root_rot=(0.257551, 0.283045, 0.683330, -0.621782),
            init_joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
        ),
        clone_in_fabric=True,
        object_init_pos=(0.0, -0.19, 0.56),
        action_scale=1.0,
        default_joint_pos_fallback={},
    ),
}


def get_allegro_hand_preset(preset_name: str) -> AllegroHandPreset:
    if preset_name not in ALLEGRO_HAND_PRESETS:
        raise ValueError(
            f"Unknown Allegro hand preset '{preset_name}'. Expected one of: {', '.join(ALLEGRO_HAND_PRESET_CHOICES)}."
        )
    return ALLEGRO_HAND_PRESETS[preset_name]


def apply_allegro_hand_preset(env_cfg: Any, preset_name: str = DEFAULT_ALLEGRO_HAND_PRESET) -> AllegroHandPreset:
    """Apply a named Allegro hand preset to the environment config."""

    preset = get_allegro_hand_preset(preset_name)
    env_cfg.hand_preset = preset_name
    env_cfg.scene.robot = preset.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.clone_in_fabric = preset.clone_in_fabric
    env_cfg.scene.object.init_state.pos = preset.object_init_pos
    env_cfg.actions.joint_pos.scale = preset.action_scale
    env_cfg.events.reset_robot_joints.params["default_joint_pos_fallback"] = dict(preset.default_joint_pos_fallback)
    return preset


ALLEGRO_HAND_CFG = get_allegro_hand_preset(DEFAULT_ALLEGRO_HAND_PRESET).robot_cfg


__all__ = [
    "ALLEGRO_HAND_CFG",
    "ALLEGRO_HAND_PRESET_CHOICES",
    "ALLEGRO_HAND_PRESETS",
    "DEFAULT_ALLEGRO_HAND_PRESET",
    "AllegroHandPreset",
    "apply_allegro_hand_preset",
    "get_allegro_hand_preset",
]
