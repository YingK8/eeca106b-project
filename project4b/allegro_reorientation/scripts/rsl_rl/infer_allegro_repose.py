# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load an RSL-RL checkpoint and run a single inference step.

This script is tailored to Template-Allegro-Reorientation-v0 and prints the expected
policy observation layout and dimensions. It uses the simulator to build the
env and network, because raw RSL-RL checkpoints require the environment to
instantiate the policy.

Observation reference (policy group, concatenated order):
  Current default (full) terms:
  1) joint_pos      : joint positions normalized to soft limits, shape (N,)
  2) object_pos     : object root position in world frame, shape (3,)
  3) object_quat    : object root orientation (w, x, y, z), shape (4,)
  4) goal_pose      : commanded pose (pos + quat), shape (7,)
  5) goal_quat_diff : quat error (asset * conj(goal)), shape (4,)
  6) last_action    : previous action sent to env, shape (N,)

  N = number of Allegro hand joints (query from robot model at runtime).
  Full total dim: 3 * N + 24

  If you switch to the NoVelObs variant, the following are removed:
    - joint_vel (N,)
    - object_lin_vel (3,)
    - object_ang_vel (3,)
  NoVelObs total dim: 2 * N + 18
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = (
    "PATH/TO/CHECKPOINT.pt"
)


parser = argparse.ArgumentParser(description="Infer a single step from an RSL-RL checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Allegro-Reorientation-v0",
    help="Name of the task (Gym ID).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint file to load.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to simulate.")
parser.add_argument(
    "--hand_preset",
    type=str,
    choices=("custom", "isaaclab"),
    default="isaaclab",
    help="Select which Allegro USD preset to use. This also applies preset-specific env defaults.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Omniverse/Isaac Sim modules must be imported after SimulationApp is instantiated.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from allegro_reorientation.assets.robots.allegro import apply_allegro_hand_preset  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "hand_preset"):
        apply_allegro_hand_preset(env_cfg, args_cli.hand_preset)

    # set log dir to checkpoint directory (used by some env utilities)
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load runner and policy (single-step inference only)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # print observation layout and dims (policy group)
    if hasattr(env.unwrapped, "observation_manager"):
        obs_mgr = env.unwrapped.observation_manager
        policy_dim = obs_mgr.group_obs_dim.get("policy", None)
        print(f"[INFO] Policy obs dim (from ObservationManager): {policy_dim}")
        print("[INFO] Policy obs terms (order preserved):")
        for name, dim in zip(obs_mgr.active_terms["policy"], obs_mgr.group_obs_term_dim["policy"]):
            print(f"  - {name}: {tuple(dim)}")
    else:
        print("[WARN] Env has no observation_manager. Falling back to observation_space.")
        print(f"[INFO] observation_space: {env.observation_space}")

    # print action dim
    if hasattr(env.unwrapped, "action_manager"):
        action_dim = env.unwrapped.action_manager.total_action_dim
        print(f"[INFO] Action dim: {action_dim}")
    else:
        print(f"[INFO] Action space: {env.action_space}")

    # run a single inference step (no env stepping)
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
    print(f"[INFO] Inference action shape: {tuple(actions.shape)}")

    # manual action processing (clamp -> rescale to limits) using provided q_min/q_max
    q_min = torch.tensor(
        [
            -0.5585,
            -0.5585,
            -0.5585,
            0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.3316,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
            -0.2792,
        ],
        device=actions.device,
        dtype=actions.dtype,
    )
    q_max = torch.tensor(
        [
            0.5585,
            0.5585,
            0.5585,
            1.5707,
            1.7278,
            1.7278,
            1.7278,
            1.1519,
            1.7278,
            1.7278,
            1.7278,
            1.7278,
            1.7278,
            1.7278,
            1.7278,
            1.7627,
        ],
        device=actions.device,
        dtype=actions.dtype,
    )
    a = actions.clamp(-1.0, 1.0)
    q_des = 0.5 * (a + 1.0) * (q_max - q_min) + q_min
    print(f"[INFO] Manual q_des shape: {tuple(q_des.shape)}")
    print(f"[INFO] Manual q_des[0]: {q_des[0].detach().cpu()}")

    # manual EMA initialization: use current joint positions (as in reset)
    robot = env.unwrapped.scene["robot"]
    prev = robot.data.joint_pos[:, : q_des.shape[1]]
    alpha = getattr(env.unwrapped.cfg.actions.joint_pos, "alpha", 0.95)
    if isinstance(alpha, torch.Tensor):
        alpha_t = alpha
    else:
        alpha_t = torch.tensor(alpha, device=actions.device, dtype=actions.dtype)
    q_cmd_manual = alpha_t * q_des + (1.0 - alpha_t) * prev
    q_cmd_manual = torch.clamp(q_cmd_manual, q_min, q_max)
    # in control loop, then set prev = q_cmd_manual, and then repeat

    # verify against action term processing (includes EMA + clamp)
    term = env.unwrapped.action_manager.get_term("joint_pos")
    term.process_actions(actions)
    q_cmd = term.processed_actions
    assert torch.allclose(q_cmd, q_cmd_manual, atol=1e-4, rtol=1e-4), "manual q_cmd != term.processed_actions"

    # pause here if you want to inspect the env before running
    obj_pos_w = obs["policy"][:, 16:19]
    robot_root_w = env.unwrapped.scene["robot"].data.root_pos_w
    env_origin = env.unwrapped.scene.env_origins

    print("obj_pos_w:", obj_pos_w[:1])
    print("obj_pos_env:", (obj_pos_w - env_origin)[:1])
    print("obj_pos_in_hand:", (obj_pos_w - robot_root_w)[:1])

    breakpoint()

    # keep the UI responsive without stepping the environment
    while simulation_app.is_running():
        simulation_app.update()

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
