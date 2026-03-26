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

        q = self.data.qpos.copy()
        n_targets = len(body_ids)
        nv = self.model.nv

        # Resolve body names to integer body ids once.
        int_ids = [mj.mj_name2id(self.model.ptr, mj.mjtObj.mjOBJ_BODY, body_name) for body_name in body_ids]

        for i in range(n_targets):
            body_id = int_ids[i]
            steps_remaining = self.max_steps

            while steps_remaining > 0:
                self.data.qpos[:] = q
                self.physics.forward()

                # Jacobian buffers are allocated per target, so index with i (not body_id).
                cur_jacp = self.jacp[i]
                cur_jacr = self.jacr[i]
                # Get the jacobian for current body
                mj.mj_jacBody(self.model.ptr, self.data.ptr, cur_jacp, cur_jacr, body_id)

                # Build a 6D task-space error to match [jacp; jacr].
                e_p = target_positions[i] - self.physics.data.xpos[body_id]
                quat_err = target_orientations[i] - self.physics.data.xquat[body_id]
                e_r = quat_err[1:]  # use xyz part so rotational error is 3D
                e = np.hstack((e_p, e_r))  # (6,)

                # constrain the error to be more  than the tolerance. If less than, we are done. 
                # e should be computed from the latest forward kinematics at every iteration. 
                if np.linalg.norm(e) < self.tol:
                    break

                J = np.vstack((cur_jacp, cur_jacr))  # (6, nv)
                JT = J.T
                # Levenberg-Marquardt damped least-squares update.
                H = JT @ J + self.damping * np.eye(nv)
                delta_q = np.linalg.solve(H, JT @ e)
                # quaternion space = 3+4 = 7d , but nv-space is only 6d (6 dof)
                # delta_q lives in nv-space; integrate into nq-space qpos.
                # delta_q lives in velocity space. 
                mj.mj_integratePos(self.model.ptr, q, self.step_size * delta_q, 1.0)
                
                # use physics parameter to clip the qpos to the valid state
                q = clip_to_valid_state(self.physics, q)
                
                steps_remaining -= 1

                ## Pseudocode for the algorithm:
                
                # goal_pose = y
                # q = current joint angles
                # step_size = desired step size
                # tolerance = set tolerance
                # e = goal_pose - current_pose
                # lambda = damping factor

                # while norm(e) >= tolerance do
                #     J = Jacobian(q)
                #     J_T = Jacobian.transpose()
                #     J_inv = (J_T * J + lambda * I).inv() * J_T
                #     delta_q = J_inv * e
                #     q += step_size * delta_q
                #     q = check_joint_limits(q)
                #     e = goal_pose - ForwardKinematics(q)
                # end while

        return q
    