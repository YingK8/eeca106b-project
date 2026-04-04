# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an RSL-RL checkpoint to TorchScript policy (.pt)."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export an RSL-RL checkpoint as TorchScript policy.")
parser.add_argument("--task", type=str, default="Template-Allegro-Reorientation-v0", help="Name of the task (Gym ID).")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to an RSL-RL checkpoint (.pt).")
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Output directory for exported policy (defaults to <checkpoint_dir>/exported).",
)
parser.add_argument("--filename", type=str, default="policy.pt", help="Exported TorchScript filename.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate.")
parser.add_argument(
    "--hand_preset",
    type=str,
    choices=("custom", "isaaclab"),
    default="isaaclab",
    help="Select which Allegro USD preset to use. This also applies preset-specific env defaults.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from allegro_reorientation.assets.robots.allegro import apply_allegro_hand_preset  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # resolve and validate checkpoint
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # configure env
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "hand_preset"):
        apply_allegro_hand_preset(env_cfg, args_cli.hand_preset)
    if args_cli.disable_fabric and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

    # set log dir to checkpoint directory (used by some env utilities)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    # create env and wrapper
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # build runner and load checkpoint
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)

    # extract policy module and optional normalizer
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export
    export_dir = args_cli.export_dir if args_cli.export_dir is not None else os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename=args_cli.filename)
    exported_path = os.path.join(export_dir, args_cli.filename)
    print(f"[INFO] Exported TorchScript policy: {exported_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
