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
        # Flatten to (16,) regardless of whether model returned (16,) or (1, 16)
        actions = actions.reshape(16).to(self._device)

        # Step 1: clamp raw policy output to [-1, 1]
        a = actions.clamp(-1.0, 1.0)

        # Step 2: rescale from [-1, 1] to joint limits
        # q_des = 0.5 * (a + 1) * (q_max - q_min) + q_min
        q_des = 0.5 * (a + 1.0) * (self._q_max - self._q_min) + self._q_min

        # Step 3: apply EMA smoothing using previous action (in sim joint order)
        # q_cmd = alpha * q_des + (1 - alpha) * prev
        q_cmd = self._alpha * q_des + (1.0 - self._alpha) * self._prev_action

        # Step 4: clamp to joint limits
        q_cmd = torch.clamp(q_cmd, self._q_min, self._q_max)

        # Step 5: update stored previous action (still in sim order)
        self._prev_action = q_cmd.clone()

        # Step 6: convert joint order from sim convention to real hardware convention
        q_cmd_real = sim2real_joints(q_cmd)

        if return_numpy:
            return q_cmd_real.cpu().numpy()
        return q_cmd_real

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


# ---------------------------------------------------------------------------
# Allegro Hand v4 FK for fingertip-to-object distances
# Matches mdp.fingertip_to_object_distances (isaaclab preset, right hand).
# Joint order (sim): index[0-3], middle[4-7], ring[8-11], thumb[12-15].
# ---------------------------------------------------------------------------

def _quat_to_rotmat_wxyz(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Hand root pose (isaaclab preset): pos=(0,0,0.5), quat (w,x,y,z)
_AH_ROOT_R = _quat_to_rotmat_wxyz(np.array([0.257551, 0.283045, 0.683330, -0.621782]))
_AH_ROOT_P = np.array([0.0, 0.0, 0.5], dtype=np.float64)

# Finger base positions in palm (hand root) frame (metres).
# Non-thumb: ±19.5 mm lateral, 149.5 mm along palm Z from wrist.
# Thumb: approximate — adjust if needed after visual inspection.
_AH_BASES = np.array([
    [ 0.0,  0.0195, 0.1495],  # index
    [ 0.0,  0.0000, 0.1495],  # middle
    [ 0.0, -0.0195, 0.1495],  # ring
    [-0.05,  0.015, 0.005 ],  # thumb (approximate)
], dtype=np.float64)

_AH_L = (0.0164, 0.054, 0.0384, 0.0267 + 0.026)  # (L0, L1, L2, L3+tip)


def _finger_fk(base: np.ndarray, j: np.ndarray) -> np.ndarray:
    """FK for one index/middle/ring finger.  J0→Rz (abduction), J1-J3→Ry (flexion)."""
    L0, L1, L2, L3 = _AH_L
    R = _rz(j[0])
    p = base + R @ np.array([0, 0, L0])
    R = R @ _ry(j[1]); p = p + R @ np.array([0, 0, L1])
    R = R @ _ry(j[2]); p = p + R @ np.array([0, 0, L2])
    R = R @ _ry(j[3]); p = p + R @ np.array([0, 0, L3])
    return p


def _thumb_fk(base: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Simplified FK for the thumb (all joints → Ry)."""
    L0, L1, L2, L3 = _AH_L
    R = _ry(j[0])
    p = base + R @ np.array([0, 0, L0])
    R = R @ _ry(j[1]); p = p + R @ np.array([0, 0, L1])
    R = R @ _ry(j[2]); p = p + R @ np.array([0, 0, L2])
    R = R @ _ry(j[3]); p = p + R @ np.array([0, 0, L3])
    return p


def allegro_fingertip_distances(q_sim: np.ndarray, cube_pos_world: np.ndarray) -> np.ndarray:
    """Approximate fingertip-to-cube distances matching the sim obs term.

    Returns (4,) float32 array in order [index, middle, ring, thumb].

    Args:
        q_sim: (16,) joint angles in sim order.
        cube_pos_world: (3,) cube position in world frame (same as policy's object_pos).
    """
    q = np.asarray(q_sim, dtype=np.float64)
    tips_palm = np.array([
        _finger_fk(_AH_BASES[0], q[0:4]),
        _finger_fk(_AH_BASES[1], q[4:8]),
        _finger_fk(_AH_BASES[2], q[8:12]),
        _thumb_fk(_AH_BASES[3], q[12:16]),
    ])
    tips_world = (_AH_ROOT_R @ tips_palm.T).T + _AH_ROOT_P
    return np.linalg.norm(tips_world - np.asarray(cube_pos_world, dtype=np.float64), axis=1).astype(np.float32)


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
