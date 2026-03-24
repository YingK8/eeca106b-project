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

        e_r = target_orientations - self.physics.data.body_xquat[body_ids] # shape (4, n)
        e_p = target_positions - self.physics.data.body_xpos[body_ids] # shape (3, n)

        e = np.vstack(e_p, e_r) # shape (7, n)

        while np.linalg.norm(e) >= self.tol and self.max_steps > 0:
            self.data.qpos[:] = q
            self.physics.forward()

            mj.mj_jacBody(self.model, self.data, self.jacp, self.jacr, body_ids[0]) 
            # see https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jacbody

            J = np.vstack((self.jacp, self.jacr)) # shape (6, nv)

            # the jacobian is the derivaitve for each manipulator body, with respect to the motor joint angles. 
            # So, you are using this to similtaneously solve for the IKs for all of the bodies!

            J_T = J.T
            J_inv = (J_T * J )
            delta_q = J_inv * e
            q += self.step_size * delta_q
            q = clip_to_valid_state(q, self.model.jnt_range)
            e_r = target_orientations - self.physics.data.body_xquat[body_ids]
            e_p = target_positions - self.physics.data.body_xpos[body_ids]
            e = np.vstack(e_p, e_r)

            self.max_steps -= 1

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
    
    