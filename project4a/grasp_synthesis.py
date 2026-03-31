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
                         max_iters=1000, 
                         lr=1,
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

            one   = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_12']
            two   = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_23']
            three = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_34']
            four  = ['ball_geom', 'sawyer/allegro_right//unnamed_geom_45']

            # Print contact status for each finger
            for idx, (name, label) in enumerate(zip([one, two, three, four], ["Finger 1", "Finger 2", "Finger 3", "Finger 4"])):
                if name in geoms:
                    print(f"{label} in contact")
                else:
                    print(f"{label} not in contact")

            # Check if all four fingertips are touching the object
            if (one in geoms and two in geoms and three in geoms and four in geoms and
                    env.physics.data.ptr.contact.frame.shape[0] >= 4):
                print("All fingers in contact with the object.")
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
                          beta=10, 
                          penetration_weight=600,
                          sync_weight=10,
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
    penetration_weight: extra multiplier for inside-sphere (negative signed distance) penalty
    sync_weight: weight on equal-rate fingertip approach penalty
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
    outside_distances = []
    d_signs = []
    for idx, p in enumerate(finger_positions):
        d = env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)
        d_signs.append(np.sign(d))
        # Print when d switches sign (penetration <-> non-penetration)
        if hasattr(joint_space_objective, "last_d_signs"):
            last = joint_space_objective.last_d_signs
            if idx < len(last) and np.sign(d) != last[idx]:
                print(f"Finger {idx+1} d switched sign at {p}: d={d:.6f}")
        # Penalize penetration more heavily so early-contact fingers do not clip through the sphere.
        if d < 0:
            print(f"Penetration detected at finger {idx+1} ({p}), d={d:.6f}, applying extra penalty.")
            surface_penalty += penetration_weight * d * d
            outside_distances.append(0.0)
        else:
            surface_penalty += d * d
            outside_distances.append(d)
    joint_space_objective.last_d_signs = d_signs

    outside_distances = np.array(outside_distances)
    sync_penalty = np.var(outside_distances)
    surface_penalty += sync_weight * sync_penalty

    # Inter-finger collision penalty
    collision_penalty = 0.0
    min_finger_dist = 0.025  # meters
    collision_weight = 100.0
    n_fingers = len(finger_positions)
    for i in range(n_fingers):
        for j in range(i+1, n_fingers):
            dist = np.linalg.norm(finger_positions[i] - finger_positions[j])
            if dist < min_finger_dist:
                print(f"Collision: Fingers {i+1} and {j+1} too close (d={dist:.4f})")
                collision_penalty += (min_finger_dist - dist) ** 2
    surface_penalty += collision_weight * collision_penalty

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
        print(f"Q+ distance: {Q_plus_dist:.6f} (eps={eps})")
        score_fc = Q_plus_dist
        if Q_plus_dist < eps:
            print(f"Switching to sufficient condition (Q-) at Q+={Q_plus_dist:.6f}")
            Q_minus_dist = optimize_sufficient_condition(G)
            print(f"Q- distance: {Q_minus_dist:.6f}")
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
    tangent_1 = normal[3:6]
    tangent_2 = normal[6:9]
    
    d_theta = 2 * np.pi / num_approx
    
    friction_cones_vecs = []
    for i in range(num_approx):
        vec =   mu * (np.cos(d_theta * i) * tangent_1 + np.sin(d_theta * i) * tangent_2) + n
        # print("Friction cone vector:", vec)
        friction_cones_vecs.append(vec)
    return friction_cones_vecs

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
    
    # define the G matrix
    G_mat = np.empty((6,0))

    # define dimension variables
    n_contacts = len(positions)  # number of contacts = number of fingers
    
    # two layer for loops to go through all the directions
    for i_ in range(n_contacts):
        pos_vec = positions[i_,:]   # (3,)
        for j_ in range(len(friction_cones[i_])):
            dir_vec = friction_cones[i_][j_] # (3,)
            G_ji = np.hstack((dir_vec, np.cross((pos_vec - origin), dir_vec))) # (6,)
            # append the current cone direction to the Grasp map matrix
            G_mat = np.hstack((G_mat, G_ji.reshape(6,1)))       # (6, N_= num_fingers * num_approx_vecs)

    return G_mat


def _generate_sphere_samples(M, seed=42):
    """
    Generate exactly M points uniformly on the 6D unit sphere S^5.
    Method: sample Gaussian vectors in R^6 and normalize each row.
    
    6 by M, so column vectors
    """
    M = int(M)
    if M <= 0:
        return np.zeros((0, 6))

    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 1.0, (6, M))
    norms = np.linalg.norm(points, axis=0)

    # Replace any zero column by a fixed nonzero vector before normalization.
    zero_mask = norms < 1e-12
    if np.any(zero_mask):
        fallback = np.zeros((6, 1))
        fallback[0, 0] = 1.0
        points[:, zero_mask] = fallback
        norms[zero_mask] = 1.0

    points_on_unit_sphere = points / norms[np.newaxis, :]

    return points_on_unit_sphere


def optimize_necessary_condition(G: np.array, *_):
    """
    Returns the result of the L2 optimization on G (Q+ distance)

    Parameters
    ----------
    G: grasp matrix
    """
    print("Optimizing necessary condition (Q+ distance)...")
    G = np.asarray(G, dtype=float)
    print(f"G shape: {G.shape}")
    print(f"G rank: {np.linalg.matrix_rank(G)}")
    print(f"G (first 3 cols):\n{G[:, :min(3, G.shape[1])].round(4)}")
    print(f"G min: {G.min():.4e}, max: {G.max():.4e}, mean: {G.mean():.4e}")
    if G.ndim != 2 or G.size == 0 or G.shape[1] == 0:
        print("Degenerate G: not enough columns or rows.")
        return float('inf')

    N = G.shape[1]
    wrench_dim = G.shape[0]
    if wrench_dim == 0:
        print("Degenerate G: zero rows.")
        return float('inf')

    # Print a warning if G is rank-deficient
    if np.linalg.matrix_rank(G) < min(G.shape):
        print("Warning: G is rank-deficient!")

    # Print alpha0 (initial guess)
    alpha0 = np.full(N, 1.0 / N)
    print(f"alpha0: {alpha0[:min(5, N)]}")

    # Numeric matrix container (Dense Matrix) for casadi
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
        {"max_iter": 500, "print_level": 0, "sb": "yes", "print_time": 0},
    )

    try:
        sol = opti.solve()
    except Exception as e:
        print(f"Q+ solver failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1e6  # Large finite penalty

    alpha_star = np.array(sol.value(alpha)).reshape(-1)
    print(f"alpha_star (first 5): {alpha_star[:min(5, len(alpha_star))]}")
    wrench_val = G @ alpha_star
    print(f"wrench (first 6): {wrench_val[:6]}")
    return float(np.linalg.norm(wrench_val))

    
def optimize_sufficient_condition(G: np.array, M=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    M: number of approximations to the norm ball

    Returns the Q- distance. Negative values indicate force-closure sufficiency.
    """
    print("Optimizing sufficient condition (Q- distance)...")
    
    G = np.asarray(G, dtype=float)
    N = G.shape[1]
    wrench_dim = G.shape[0]
    Q = _generate_sphere_samples(M)

    c = np.zeros(N + 1)
    c[-1] = -1.0
    bounds = [(0.0, None)] * N + [(0.0, None)]

    d_q_vals = []

    # LP in variables [alpha_1 ... alpha_N r] for each sampled direction q_k.
    for k in range(M):
        qk = Q[:, k]

        A_eq = np.zeros((wrench_dim + 1, N + 1))
        A_eq[:wrench_dim, :N] = G
        A_eq[:wrench_dim, -1] = -qk
        A_eq[wrench_dim, :N] = 1.0

        b_eq = np.zeros(wrench_dim + 1)
        b_eq[wrench_dim] = 1.0

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            d_q_vals.append(float(res.x[-1]))

    if not d_q_vals:
        return 0.0

    return -min(d_q_vals)
