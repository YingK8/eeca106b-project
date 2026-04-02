import numpy as np
import dm_control
import types
import AllegroHandEnv

def clip_to_valid_state(physics: dm_control.mjcf.physics.Physics, qpos: np.array):
    """
    This function returns qpos with every value clipped to the allowable joint range as specified
    in the MJCF. 
    """
    qpos_clipped = qpos.copy()

    for joint_idx in range(physics.model.njnt):
        joint_range = physics.model.jnt_range[joint_idx]

        qpos_clipped[physics.model.jnt_qposadr[joint_idx]] = np.clip(
            qpos_clipped[physics.model.jnt_qposadr[joint_idx]], 
            joint_range[0],
            joint_range[1])

    return qpos_clipped

# [MT]
def clip_to_valid_state_slice(physics: dm_control.mjcf.physics.Physics, qpos: np.array, q_slice: slice=None):
    """
    This function returns qpos with every value clipped to the allowable joint range as specified
    in the MJCF, allowing user to specify a slice
    """
    
    if not q_slice:
        qpos_clipped = qpos.copy()
        for joint_idx in range(physics.model.njnt):
            joint_range = physics.model.jnt_range[joint_idx]

            qpos_clipped[physics.model.jnt_qposadr[joint_idx]] = np.clip(
                qpos_clipped[physics.model.jnt_qposadr[joint_idx]], 
                joint_range[0],
                joint_range[1])
    else:
        # if a slice of the model joints are specified
        qpos_full           = physics.data.qpos.copy()
        qpos_full[q_slice]  = qpos.copy()
        for joint_idx in range(physics.model.njnt):
            joint_range = physics.model.jnt_range[joint_idx]

            qpos_full[physics.model.jnt_qposadr[joint_idx]] = np.clip(
                qpos_full[physics.model.jnt_qposadr[joint_idx]], 
                joint_range[0],
                joint_range[1])
        qpos_clipped = qpos_full[q_slice]

    return qpos_clipped

def numeric_gradient(function: types.FunctionType, 
                     q_h: np.array, 
                     env: AllegroHandEnv, 
                     fingertip_names: list[str], 
                     in_contact: bool, 
                     eps=0.01):
    """
    This function approximates the gradient of the joint_space_objective

    Parameters
    ----------
    function: function we are taking the gradient of
    q_h: joint configuration of the hand 
    env: AllegroHandEnv instance 
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    eps: hyperparameter for the delta of the gradient 

    Output
    ------
    Approximate gradient of the inputted function
    """
    # baseline = function(q_h, env, fingertip_names, in_contact)  # original `starter`
    baseline = function(env, q_h, fingertip_names, in_contact)  # updated `starter`
    grad = np.zeros_like(q_h)
    for i in range(len(q_h)):
        q_h_pert = q_h.copy()
        q_h_pert[i] += eps
        # val_pert = function(q_h_pert, env, fingertip_names, in_contact) # original `starter`
        val_pert = function(env, q_h_pert, fingertip_names, in_contact) # updated `starter`
        grad[i] = (val_pert - baseline) / eps
    return grad

def quaternion_error_naive(current_quat: np.array, target_quat: np.array):
    """
    Rough orientation error between two quaternions.
    This is just a rough measure and doesn't work well for
    large angles, so you might want to consider using
    something more advanced to compare quaternions.
    """
    q_diff = quat_multiply(target_quat, quat_conjugate(current_quat))
    # Ignore w
    return q_diff[1:4]

def quat_multiply(q1: np.array, q2: np.array):
    """Multiply two quaternions, returning q1*q2 in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def quat_conjugate(q: np.array):
    """Return quaternion conjugate: [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

# [MT]
def sampleRandomPointOnHypersphere(dim: int, 
                                   num_pts: int,
                                   rng = None,
                                   ) -> np.ndarray:
    '''
    Muller (1959) and Marsaglia (1972)
    Random point on a hypersphere S^{n-1}
    Sample an n-D standard normal vector then normalize
    Input:
        dim:        int, dimension of the Euclidean vector, Isomorphic to S^{n-1}
        num_pts:    int, number of random points to be generated
        rng:        random number generator,
    '''
    rng = np.random.default_rng(rng)
    Vec = rng.normal(size = (num_pts, dim))
    return Vec / np.linalg.norm(Vec, axis=1, keepdims=True)

