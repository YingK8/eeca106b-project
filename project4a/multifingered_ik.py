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
        body_ids: list of length n containing the ids for every body

        Returns
        -------
        new_qpos: np.array of size self.physics.data.qpos containing desired positions in joint space

        Tips: 
            -To access the body id you can use: self.model.body([insert name of body]).id 
            -You should consider using clip_to_valid_state in utils.py to ensure that joint poisitons
            are possible 
        """

        # TODO(1): Normalize / validate inputs.
        # - Convert target_positions and target_orientations to np.ndarray.
        # - Enforce shapes:
        #     target_positions:      (n_targets, 3)
        #     target_orientations:   (n_targets, 4) in [w, x, y, z]
        # - Ensure len(body_ids) == n_targets.
        # - If your class doc says 3xn / 4xn, decide on one convention and convert once here.



        # TODO(2): Resolve body references to integer body ids.
        # - body_ids may be strings (body names) or already-int ids.
        # - If strings, resolve with: self.model.body(name).id
        # - Keep a list/array: resolved_body_ids of length n_targets.

        # TODO(3): Initialize optimization state.
        # - q: working copy of current generalized coordinates.
        # - nv: number of generalized velocities (dimension of Jacobian columns).
        # - Use class params already provided in __init__:
        #       self.step_size, self.tol, self.damping, self.max_steps, self.alpha
        # - Typical shapes used below:
        #       residual r: (6 * n_targets,)
        #       Jacobian J: (6 * n_targets, nv)
        q = self.data.qpos.copy()
        n_targets = len(body_ids)
        nv = self.model.nv

        for step in range(self.max_steps):
            

        # TODO(4): Main Levenberg-Marquardt loop.
        # for step in range(self.max_steps):
        #   a) Write q into simulator state and run forward kinematics:
        #        self.data.qpos[:] = q
        #        self.physics.forward()
        #
        #   b) Build stacked residual vector r and Jacobian J:
        #        For each target i:
        #          - current_pos  = self.data.xpos[body_id]
        #          - current_quat = self.data.xquat[body_id]
        #          - pos_err = target_positions[i] - current_pos                (3,)
        #          - rot_err = quaternion_error_naive(current_quat, target_quat) (3,)
        #          - r_i = [pos_err, alpha * rot_err]                            (6,)
        #
        #          Compute body Jacobians at current q:
        #            mj.mj_jacBody(self.model.ptr, self.data.ptr, jacp_i, jacr_i, body_id)
        #          where jacp_i, jacr_i have shape (3, nv).
        #
        #          J_i = [jacp_i;
        #                 alpha * jacr_i]                                        (6, nv)
        #          Insert J_i into J block for target i.
        #
        #   c) Check convergence using residual norm:
        #        if np.linalg.norm(r) < self.tol: break
        #
        #   d) Levenberg-Marquardt update:
        #        Solve (J^T J + lambda * I) * delta_q = J^T r
        #        with lambda = self.damping.
        #        Use np.linalg.solve if matrix is well-conditioned;
        #        otherwise fallback to np.linalg.lstsq.
        #
        #   e) Apply step and joint-limit projection:
        #        q_candidate = q + self.step_size * delta_q
        #        q_candidate = clip_to_valid_state(self.physics, q_candidate)
        #
        #   f) Optional LM damping adaptation (recommended):
        #        - Evaluate candidate error norm.
        #        - If error improved: accept q_candidate, decrease lambda.
        #        - Else: reject candidate, increase lambda.

        # TODO(5): Finalize state and return q.
        # - Set simulator state to final q and forward once if needed.
        # - Return q with same shape as self.data.qpos.

        raise NotImplementedError(
            "TODO: Implement Levenberg-Marquardt IK using the scaffold in calculate()."
        )
    
    