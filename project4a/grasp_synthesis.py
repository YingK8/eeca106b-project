import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
from utils import *

import casadi as ca

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: AllegroHandEnv.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=2000, 
                         lr=0.5,
                         return_history=False):
    """
    Given an initial hand joint configuration, q_h_init, return adjusted joint angles that are touching
    the object and approximate force closure. This is algorithm 1 in the project specification.

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h_init: array of joint positions for the hand
    max_iters: maximum number of iterations for the optimization
    lr: learning rate for the gradient step

    Output
    ------
    New joint angles after contact and force closure adjustment
    """
    q_h = q_h_init.copy()
    history = [q_h.copy()] if return_history else None
    for it in range(max_iters):
        in_contact = False
        if env.physics.data.ncon >= 4:
            geom_id_pairs = env.physics.data.ptr.contact.geom
            # Get list of contact pair geoms
            geoms = np.array([
                [
                    mj.mj_id2name(env.physics.model.ptr, mj.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in pair
                ]
                for pair in geom_id_pairs
            ])

            # Specify the exact geom contacts we're looking for (if you change the allegro hand urdf you might want to check that these still correspond to the fingertips)
            one   = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_12']
            two   = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_23']
            three = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_34']
            four  = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_45']

            # Check if all four fingertips are touching the object
            if (one in geoms and two in geoms and three in geoms and four in geoms and
                    env.physics.data.ptr.contact.frame.shape[0] >= 4):
                in_contact = True
        
        # Evaluate the objective function and check its gradient
        fval = joint_space_objective(env, q_h, fingertip_names, in_contact)
        grad = numeric_gradient(joint_space_objective, q_h, env, fingertip_names, in_contact)

        # Update the joint configuration
        q_h_new = q_h.copy() - lr*grad
        
        # Clip joint configuration to be in bounds
        qpos_new = env.physics.data.qpos.copy()
        qpos_new[env.q_h_slice] = q_h_new
        qpos_new = clip_to_valid_state(env.physics, qpos_new)
        q_h_new = qpos_new[env.q_h_slice].copy()

        # Evaluate the objective function with the new joint configuration to measure improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact)

        # Only update q_h if the objective function has improved
        if fval_new < fval:
            q_h = q_h_new
            if return_history:
                history.append(q_h.copy())
            improvement = fval - fval_new 
            print(f"Iter {it}, objective={fval_new:.4f}, improvement={improvement:.4f}")
            if improvement < 1e-6:
                break
        else:
            # If no improvement, reduce lr or break
            lr *= 0.5
            print(f"Iter {it}, no improvement, reduce lr to {lr}")
            if lr < 1e-6:
                break
    if return_history:
        return q_h, history
    return q_h

def joint_space_objective(env: AllegroHandEnv.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=5, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=8,
                          eps=0.00001):
    """
    This function minimizes an objective such that the distance from the origin
    in wrench space as well as distance from fingers to object surface is minimized.
    This is algorithm 2 in the project specification. 

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h: array of joint positions for the hand
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    beta: weight coefficient on the surface penalty 
    friction_coeff: Friction coefficient for the ball
    num_friction_cone_approx: number of approximation vectors in the friction cone
    
    Output
    ------
    fc_loss + (beta * d) as written in algorithm 2
    """
    env.set_configuration(q_h)
    finger_positions = env.get_body_positions(fingertip_names)

    # Penalty for distance from surface
    surface_penalty = 0.0
    for p in finger_positions:
        d = env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)
        surface_penalty += d*d
    if not in_contact or env.physics.data.ptr.contact.frame.shape[0] < 4:
        return beta * surface_penalty
    else: # Fingers are in contact, so we calculate Q+ and Q- penalty
        # Create friction cone
        contact_frames, contact_positions = env.get_contact_normals_and_positions(env.physics.data.ptr.contact)
        directions_list = []
        for i in range(len(contact_frames)):
            contact_frame = contact_frames[i]
            directions_i = build_friction_cone(contact_frame, friction_coeff, num_friction_cone_approx)
            directions_list.append(directions_i)

        G = build_grasp_matrix(contact_positions, directions_list, origin=env.sphere_center)

        # First optimize Q+ distance until it's near zero, then switch to optimizing Q- distance
        Q_plus_dist = optimize_necessary_condition(G, env)
        if Q_plus_dist > eps:
            score_fc = Q_plus_dist
        else:
            Q_minus_dist = optimize_sufficient_condition(G)
            score_fc = Q_minus_dist
        return score_fc + beta * surface_penalty
    
def build_friction_cone(normal: np.array, mu=0.5, num_approx=4):
    """
    This function builds a discrete friction cone around each normal vector. 

    Parameters
    ----------
    normal: (,9) np.array containing the normal and tangent directions of the contact
    mu: friction coefficient
    num_approx: number of approximation vectors in the friction cone

    Output
    ------
    friction_cone_vectors: array of discretized friction cone vectors around the given normal
    """
    n = normal[0:3]
    n = n / max(np.linalg.norm(n), 1e-12)
    tangent_1 = normal[3:6]
    tangent_2 = normal[6:9]
    
    friction_cones_vecs = []
    for angle in np.linspace(0.0, 2.0 * np.pi, num_approx, endpoint=False):
        # Tangential component
        tangent_f = np.cos(angle) * tangent_1 + np.sin(angle) * tangent_2
        # Friction cone force: normal + scaled tangent
        f = n + mu * tangent_f
        friction_cones_vecs.append(f / np.linalg.norm(f))
    return np.array(friction_cones_vecs)


def build_grasp_matrix(positions: np.array, friction_cones: list, origin=np.zeros(3)):
    """
    Builds a grasp map containing wrenches along the discretized friction cones. 

    Parameters
    ----------
    positions: nx3 np.array of contact positions where n is the number of contacts
    firction_cone: a list of lists as outputted by build_friction_cones. 
    origin: the torque reference. In this case, it's the object center.
    
    Return a 2D numpy array G with shape (6, number_of_cone_directions).
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    origin = np.asarray(origin, dtype=float).reshape(3,)

    wrench_columns = []
    for p, cone_dirs in zip(positions, friction_cones):
        r = p - origin
        for f in np.atleast_2d(np.asarray(cone_dirs, dtype=float)):
            wrench_columns.append(np.hstack((f, np.cross(r, f))))

    if not wrench_columns:
        return np.zeros((6, 0))
    return np.column_stack(wrench_columns)


def _generate_sphere_samples(M, seed=42):
    """
    Generate exactly M points uniformly on the 6D unit sphere S^5.
    Method: sample Gaussian vectors in R^6 and normalize each row.
    """
    M = int(M)
    if M <= 0:
        return np.zeros((0, 6))

    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 1.0, (M, 6))
    norms = np.linalg.norm(points, axis=1)

    # Extremely unlikely, but prevent divide-by-zero if a near-zero vector appears.
    zero_mask = norms < 1e-12
    while np.any(zero_mask):
        points[zero_mask] = rng.normal(0.0, 1.0, (np.sum(zero_mask), 6))
        norms = np.linalg.norm(points, axis=1)
        zero_mask = norms < 1e-12

    points_on_unit_sphere = points / norms[:, np.newaxis]

    return points_on_unit_sphere  

def optimize_necessary_condition(G: np.array, *_):
    """
    Returns the result of the L2 optimization on G (Q+ distance)

    Parameters
    ----------
    G: grasp matrix
    """
    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.size == 0 or G.shape[1] == 0:
        return float('inf')

    N = G.shape[1]
    wrench_dim = G.shape[0]
    if wrench_dim == 0:
        return float('inf')

    G_ca = ca.DM(G)

    opti = ca.Opti()
    alpha = opti.variable(N, 1)
    wrench = G_ca @ alpha

    opti.minimize(ca.sumsqr(wrench))
    opti.subject_to(ca.sum1(alpha) == 1)
    opti.subject_to(alpha >= 0)

    opti.solver(
        "ipopt",
        {"expand": True},
        {"max_iter": 500, "print_level": 0, "sb": "yes"},
    )

    try:
        sol = opti.solve()
    except Exception:
        return float('inf')

    alpha_star = np.array(sol.value(alpha)).reshape(-1)
    return float(np.linalg.norm(G @ alpha_star))
    
    

def optimize_sufficient_condition(G: np.array, M=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    M: number of approximations to the norm ball

    Returns the Q- distance. Negative values indicate force-closure sufficiency.
    """

    G = np.asarray(G, dtype=float)

    Q = _generate_sphere_samples(M)
    N = G.shape[1]
    G_ca = ca.DM(G)

    d_k_values = []
    for q_k in Q:
        opti = ca.Opti()

        alpha = opti.variable(N, 1)
        r = opti.variable()
        q_ca = ca.DM(q_k).reshape((6, 1))

        # max r s.t. G alpha = r q_k, alpha in simplex, r >= 0.
        # Equivalent NLP for IPOPT: min -r.
        opti.minimize(-r)
        opti.subject_to(G_ca @ alpha == r * q_ca)
        opti.subject_to(ca.sum1(alpha) == 1)
        opti.subject_to(alpha >= 0)
        opti.subject_to(r >= 0)

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 500, "print_level": 0, "sb": "yes"},
        )

        # optimization problem may not be always well constructed, using try-except
    try:
        sol = opti.solve()
        return -float(sol.value(r)) # notice the negative sign
    except Exception as err:
        print(err)
    
    return 0.0  # return 0.0 radius if no valid solution
