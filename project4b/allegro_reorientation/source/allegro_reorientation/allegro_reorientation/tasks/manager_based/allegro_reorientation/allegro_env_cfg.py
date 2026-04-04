# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import inhand_env_cfg

##
# Pre-defined configs
##
from allegro_reorientation.assets.robots.allegro import (  # isort: skip
    DEFAULT_ALLEGRO_HAND_PRESET,
    apply_allegro_hand_preset,
)


@configclass
class AllegroCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    hand_preset: str = DEFAULT_ALLEGRO_HAND_PRESET

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Apply the default hand preset during config construction. The train/play
        # scripts can override this later from the command line.
        apply_allegro_hand_preset(self, self.hand_preset)


@configclass
class AllegroCubeEnvCfg_PLAY(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None


##
# Environment configuration with no velocity observations.
##


@configclass
class AllegroCubeNoVelObsEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = inhand_env_cfg.ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class AllegroCubeNoVelObsEnvCfg_PLAY(AllegroCubeNoVelObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
