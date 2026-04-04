"""Simplified policy wrapper for real-world Allegro (single-env, 16-DoF only)."""


from collections.abc import Mapping
from typing import Any

import numpy as np
import torch



class RslRlPolicyWrapperRealWorld:
    """TorchScript policy wrapper for single-env Allegro (16-DoF only)."""

    def __init__(
        self,
        model_path: str,
        device: str,
        q_min: np.ndarray | torch.Tensor,
        q_max: np.ndarray | torch.Tensor,
        alpha: float,
        action_scale: float,
    ) -> None:
        self._device = torch.device(device)
        self._model = torch.jit.load(model_path, map_location=self._device)
        self._model.eval()
        assert hasattr(self._model, "reset"), "TorchScript policy must expose reset()"

        self._q_min = self._as_tensor_value(q_min, self._device)
        self._q_max = self._as_tensor_value(q_max, self._device)
        assert self._q_min.shape == (16,)
        assert self._q_max.shape == (16,)
        assert 0.0 <= alpha <= 1.0
        self._alpha = torch.tensor(alpha, device=self._device, dtype=torch.float32)
        self.action_scale = action_scale

        self._prev_action = torch.zeros(16, device=self._device, dtype=torch.float32)

    def reset(self) -> None:
        self._model.reset()

    def set_prev_action(self, prev_action: np.ndarray | torch.Tensor) -> None:
        prev = self._as_tensor_value(prev_action, self._device)
        assert prev.shape == (16,)
        self._prev_action = prev

    def act(self, obs: Any, return_numpy: bool = True):
        obs_tensor = self._to_tensor(obs)
        with torch.inference_mode():
            raw_actions = self._model(obs_tensor)
        q_cmd = self._postprocess(raw_actions.clone(), return_numpy)
        return raw_actions, q_cmd

    __call__ = act

    def _to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, Mapping):
            assert "policy" in obs, "Expected obs dict with 'policy'"
            obs = obs["policy"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        if obs.ndim == 2:
            assert obs.shape[0] == 1, "Expected single-env batch"
            obs = obs.squeeze(0)
        assert obs.ndim == 1, "Expected 1D obs"
        return obs.to(self._device)

    def _postprocess(self, actions: torch.Tensor, return_numpy: bool):
        # TODO: implement policy output postprocessing for real hardware.
        # Hint: the policy output is in simulation convention. Before sending commands
        # to the real hand, think through whether simulation and real hardware have different joint order.
        raise NotImplementedError("TODO: implement real-world postprocessing for policy actions")

    def _as_tensor_value(self, value: np.ndarray | torch.Tensor | float, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, device=device, dtype=torch.float32)


def sim2real_joints(q_sim: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert joint order from sim -> real.
    Input shape: (16,)
    """
    if isinstance(q_sim, torch.Tensor):
        assert q_sim.shape == (16,)
        q = q_sim.reshape(4, 4)
        q = q.T
        return q.reshape(16)
    q_sim = np.asarray(q_sim)
    assert q_sim.shape == (16,)
    q = q_sim.reshape(4, 4)
    q = q.T
    return q.reshape(16)


def real2sim_joints(q_real: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert joint order from real -> sim.
    Input shape: (16,)
    """
    if isinstance(q_real, torch.Tensor):
        assert q_real.shape == (16,)
        q = q_real.reshape(4, 4)
        q = q.T
        return q.reshape(16)
    q_real = np.asarray(q_real)
    assert q_real.shape == (16,)
    q = q_real.reshape(4, 4)
    q = q.T
    return q.reshape(16)

def quat_mul(q1, q2):
    # (w,x,y,z)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)

def quat_diff_angle(q_diff):
    w = q_diff[0]
    w = max(-1.0, min(1.0, abs(w)))
    return 2.0 * np.arccos(w)

def goal_quat_diff(object_quat, goal_quat, make_quat_unique=False):
    # both in world frame, (w,x,y,z)
    q = quat_mul(object_quat, quat_conjugate(goal_quat))
    if make_quat_unique and q[0] < 0:
        q = -q
    diff_angle = quat_diff_angle(q)
    return q, diff_angle


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half)*axis)], dtype=np.float32)


def quat_unique(q):
    # ensure w >= 0
    return q if q[0] >= 0 else -q


def generate_goal_pose(
    default_pos=(0.0, -0.19, 0.56),
    init_pos_offset=(0.0, 0.0, -0.04),
    make_quat_unique=False,
    rng=None,
):
    """
    Returns (pos, quat) where:
      pos is env-frame position (3,)
      quat is world-frame orientation (w,x,y,z)
    """
    if rng is None:
        rng = np.random.default_rng()


    pos = np.array(default_pos, dtype=np.float32) + np.array(init_pos_offset, dtype=np.float32)


    # random rotations about X and Y
    r = rng.uniform(-1.0, 1.0, size=(2,))
    qx = quat_from_angle_axis(r[0] * np.pi, [1,0,0])
    qy = quat_from_angle_axis(r[1] * np.pi, [0,1,0])
    quat = quat_mul(qx, qy)


    if make_quat_unique:
        quat = quat_unique(quat)


    return pos, quat
