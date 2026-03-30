import numpy as np
import mujoco as mj
import dm_control
from utils import *

# Inspired by: https://alefram.github.io/posts/Basic-inverse-kinematics-in-Mujoco
class LevenbergMarquardtIK:
    
    def __init__(self, model: dm_control.mujoco.wrapper.core.MjModel, 
                 data: dm_control.mujoco.wrapper.core.MjData, 
                 step_size: int, 
                 tol: int, 
                 alpha: int, 
                 jacp: np.array, 
                 jacr: np.array, 
                 damping: int, 
                 max_steps: int,
                 physics: dm_control.mjcf.physics.Physics):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
        self.max_steps = max_steps
        self.physics = physics

        # Cache hand joint indices for clipping only the Allegro hand configuration.
        self._hand_joint_meta = self._infer_hand_joint_meta()

    def _infer_hand_joint_meta(self):
        """Infer Allegro hand hinge/slide joints and their qpos addresses."""
        hand_joint_meta = []
        name_tokens = ("/ffj", "/mfj", "/rfj", "/thj", "ffj", "mfj", "rfj", "thj")

        for joint_id in range(self.model.njnt):
            joint_name = mj.mj_id2name(self.model.ptr, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None or not any(token in joint_name for token in name_tokens):
                continue

            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue

            qpos_adr = self.model.jnt_qposadr[joint_id]
            joint_range = self.model.jnt_range[joint_id]
            hand_joint_meta.append((qpos_adr, joint_range[0], joint_range[1]))

        return hand_joint_meta

    def _clip_hand_joints(self, qpos):
        """Clip only Allegro hand joints to their valid limits."""
        qpos_clipped = qpos.copy()
        for qpos_adr, lower, upper in self._hand_joint_meta:
            qpos_clipped[qpos_adr] = np.clip(qpos_clipped[qpos_adr], lower, upper)
        return qpos_clipped

    def _orientation_error(self, current_quat, target_quat):
        """Quaternion-relative orientation error in 3D (xyz part)."""
        current = current_quat / (np.linalg.norm(current_quat) + 1e-12)
        target = target_quat / (np.linalg.norm(target_quat) + 1e-12)

        q_err = quat_multiply(target, quat_conjugate(current))
        if q_err[0] < 0.0:
            q_err = -q_err
        return q_err[1:4]
    
    def calculate(self, target_positions: np.array, 
              target_orientations: np.array, 
              body_ids: list, 
              evaluating=False):
        """
        Calculates joint angles given target positions and orientations by solving inverse kinematics.
        Uses the Levenberg-Marquardt method for nonlinear optimization. 

        Parameters
        ----------
        target_positions: 3xn np.array containing n desired x,y,z positions
        target_orientations: 4xn np.array containing n desired quaternion orientations
        body_ids: list of length n containing body names

        Returns
        -------
        q: np.array of size self.physics.data.qpos containing desired positions in joint space

        Algorithm:
            For each target body, minimize the task-space error using Levenberg-Marquardt damping.
        """
        
        # --- Initialization (goal_pose, q, step_size, tolerance, lambda) ---
        q = self.data.qpos.copy()
        target_positions = np.asarray(target_positions, dtype=float)
        target_orientations = np.asarray(target_orientations, dtype=float)

        n_targets = len(body_ids)
        if target_positions.shape[0] != n_targets or target_orientations.shape[0] != n_targets:
            raise ValueError("target_positions, target_orientations, and body_ids must have matching first dimension")

        nv = self.model.nv
        alpha = float(self.alpha)
        step_size = float(self.step_size)
        tolerance = float(self.tol)
        lambda_damping = float(self.damping)
        max_steps = int(self.max_steps)

        # Resolve body names once before the iterative solve.
        int_body_ids = []
        for name in body_ids:
            body_id = mj.mj_name2id(self.model.ptr, mj.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Body name not found in model: {name}")
            int_body_ids.append(body_id)

        # --- Levenberg-Marquardt loop: while norm(e) >= tolerance ---
        for _ in range(max_steps):
            # Forward kinematics at current q.
            self.data.qpos[:] = q
            self.physics.forward()

            # Build stacked task-space error e and stacked Jacobian J.
            e = np.zeros(6 * n_targets)
            J = np.zeros((6 * n_targets, nv))

            for i, body_id in enumerate(int_body_ids):
                row0 = 6 * i
                row1 = row0 + 6

                current_position = self.physics.data.xpos[body_id]
                current_orientation = self.physics.data.xquat[body_id]
                goal_position = target_positions[i]
                goal_orientation = target_orientations[i]

                pos_err = goal_position - current_position
                ori_err = self._orientation_error(current_orientation, goal_orientation)
                e[row0:row1] = np.hstack((pos_err, ori_err))

                jac_position = self.jacp[i]
                jac_orientation = self.jacr[i]
                mj.mj_jacBody(self.model.ptr, self.data.ptr, jac_position, jac_orientation, body_id)
                J[row0:row1, :] = np.vstack((jac_position, jac_orientation))

            # Stop when the task-space error norm reaches tolerance.
            if np.linalg.norm(e) < tolerance:
                break

            # Damped least-squares inverse:
            # delta_q = (J^T J + lambda I)^(-1) J^T e
            J_T = J.T
            H = J_T @ J + lambda_damping * np.eye(nv)
            rhs = J_T @ e

            try:
                delta_q = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                lambda_damping *= 10.0
                delta_q = np.linalg.pinv(H) @ rhs

            # q += step_size * delta_q, then enforce joint limits.
            mj.mj_integratePos(self.model.ptr, q, alpha * step_size * delta_q, 1.0)
            q = self._clip_hand_joints(q)

        return q
    