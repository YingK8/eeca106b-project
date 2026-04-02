import numpy as np
import mujoco as mj
import dm_control
from utils import *

from scipy.spatial.transform import Rotation as R



DEBUG_FLAG = True


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
        # Initialize IK solver
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp        # [num_components, 3, N]
        self.jacr = jacr        # [num_components, 3, N]
        self.damping = damping
        self.max_steps = max_steps
        self.physics = physics
    
    def calculate(self, target_positions: np.array, 
                  target_orientations: np.array, 
                  body_ids: list, 
                  evaluating=False):
        # Solve IK for given target positions and orientations

        # Check input dimensions
        if (len(target_positions) != len(target_orientations)) or (len(target_orientations) != len(body_ids)):
            raise ValueError("Input dimension for `target_positions`, `target_orientations` and `body_ids` do not match")

        # Setup variables
        iter_cnt = 0
        num_poses = len(body_ids)               # number of desired poses

        # Forward compute MuJoCo graph
        mj.mj_forward(self.model.ptr, self.data.ptr)
        # Compute error vector
        error = np.zeros(len(target_positions) * 6) # 6 workspace DOF for each robot component

        # Compute position error
        current_pos = np.hstack([self.data.body(key_).xpos for key_ in body_ids])
        error = np.subtract(np.hstack([vals for vals in target_positions]), current_pos)

        # Compute orientation error
        ori_errs = np.hstack([quaternion_error_naive(self.data.body(id_).xquat, target_orientations[i_]) for i_, id_ in enumerate(body_ids)])
        error = np.hstack((error, ori_errs))

        # Setup Jacobians
        N_          = self.jacp[0,:,:].shape[1] # number of variables
        Identity    = np.identity(N_)           # identity matrix for LM update
        jacp_cur    = np.zeros((num_poses * 3, N_))         # positional Jacobian (numPose * 3, N)
        jacr_cur    = np.zeros((num_poses * 3, N_))         # rotational Jacobian (numPose * 3, N)
        stacked_jac = np.zeros((num_poses * 6, N_))         # stacked Jacobian    (numPose * 6, N)

        # Iterative update loop
        while (np.linalg.norm(error) >= self.tol and iter_cnt <= self.max_steps):
            # increment counter
            iter_cnt += 1

            # Compute Jacobian
            for id_ in range(num_poses):
                # compute Jacobian `mj_jac`
                mj.mj_jac(self.model.ptr, self.data.ptr,
                      self.jacp[id_,:,:],                   # (3, N)
                      self.jacr[id_,:,:],                   # (3, N)
                      target_positions[id_],
                      self.model.body(body_ids[id_]).id
                      )
                # Stack Jacobian
                jacp_cur[id_*3:(id_+1)*3, :] = self.jacp[id_,:,:] # (3, N)
                jacr_cur[id_*3:(id_+1)*3, :] = self.jacr[id_,:,:] # (3, N)

            # Build stacked Jacobian and LM Hessian
            stacked_jac = np.vstack((jacp_cur, jacr_cur))
            product     = stacked_jac.T @ stacked_jac + self.damping * Identity

            # Compute pseudo-inverse
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ stacked_jac.T
            else:
                j_inv = np.linalg.inv(product) @ stacked_jac.T
            
            # Compute update step
            delta_q = j_inv @ error
            
            # Update joint configs
            self.data.qpos[7:] += self.step_size * delta_q[6:]  # first 7 dimensions refer to Ball's joint
            # [TODO]: check correctness for ball rotational update
            self.data.qpos[0:3] += self.step_size * delta_q[0:3]
            self.data.qpos[3:7] = (R.from_quat(self.data.qpos[3:7],scalar_first=True) \
                                   * R.from_rotvec(delta_q[3:6])
                                   ).as_quat(scalar_first=True)
            # Clip to valid joint limits
            self.data.qpos      = clip_to_valid_state(self.physics, self.data.qpos)
            # Forward physics
            mj.mj_forward(self.model.ptr, self.data.ptr)
            # Compute new error
            current_pos = np.hstack([self.data.body(key_).xpos for key_ in body_ids])
            error = np.subtract(np.hstack([vals for vals in target_positions]), current_pos)
            # Orientation error
            ori_errs = np.hstack([quaternion_error_naive(self.data.body(id_).xquat, target_orientations[i_]) for i_, id_ in enumerate(body_ids)])
            error = np.hstack((error, ori_errs))


        if DEBUG_FLAG:
            for id_ in range(len(body_ids)):
                print("final pos of {0} is: ".format(id_), self.data.body(body_ids[id_]).xpos )
                print("final ori of {0} is: ".format(id_), self.data.body(body_ids[id_]).xquat)
            print("final error is: ", np.linalg.norm(error))
            print("number of iterations: ", iter_cnt)


        # Return final joint configuration
        return self.data.qpos








if __name__ == "__main__":
    main()
    
    