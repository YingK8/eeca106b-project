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
        qadr = physics.model.jnt_qposadr[joint_idx]
        joint_type = int(physics.model.jnt_type[joint_idx])

        # MuJoCo joint type ids: free=0, ball=1, slide=2, hinge=3.
        if joint_type in (2, 3):
            if bool(physics.model.jnt_limited[joint_idx]):
                joint_range = physics.model.jnt_range[joint_idx]
                qpos_clipped[qadr] = np.clip(qpos_clipped[qadr], joint_range[0], joint_range[1])
        elif joint_type == 0:
            quat = qpos_clipped[qadr + 3 : qadr + 7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-12:
                qpos_clipped[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                qpos_clipped[qadr + 3 : qadr + 7] = quat / quat_norm
        elif joint_type == 1:
            quat = qpos_clipped[qadr : qadr + 4]
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-12:
                qpos_clipped[qadr : qadr + 4] = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                qpos_clipped[qadr : qadr + 4] = quat / quat_norm

    return qpos_clipped

def numeric_gradient(function: types.FunctionType,
                     q_h: np.array,
                     env: AllegroHandEnv,
                     fingertip_names: list[str],
                     in_contact: bool,
                     beta=None,
                     eps=1e-3):
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
    if beta is None:
        baseline = function(env, q_h, fingertip_names, in_contact)
    else:
        baseline = function(env, q_h, fingertip_names, in_contact, beta=beta)

    if not np.isfinite(baseline):
        return np.zeros_like(q_h)

    grad = np.zeros_like(q_h)
    for i in range(len(q_h)):
        q_h_pert = q_h.copy()
        q_h_pert[i] += eps
        if beta is None:
            val_pert = function(env, q_h_pert, fingertip_names, in_contact)
        else:
            val_pert = function(env, q_h_pert, fingertip_names, in_contact, beta=beta)

        if not np.isfinite(val_pert):
            grad[i] = 0.0
        else:
            grad[i] = (val_pert - baseline) / eps

    grad[~np.isfinite(grad)] = 0.0
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