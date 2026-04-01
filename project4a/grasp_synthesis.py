import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
from utils import *

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: AllegroHandEnv.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=1000, 
                         lr=5,
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
    threshold = 0.01  # meters, for fingertip proximity
    max_iters = 300
    beta = 50
    min_lr = 1e-6
    initial_lr = lr
    q_plus_last = float("inf")
    q_minus_last = float("inf")
    for it in range(max_iters):
        # Contact detection diagnostics
        in_contact = False
        contact_fingers = [False, False, False, False]
        if env.physics.data.ncon >= 4:
            geom_id_pairs = env.physics.data.ptr.contact.geom
            geoms = np.array([
                [
                    mj.mj_id2name(env.physics.model.ptr, mj.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in pair
                ]
                for pair in geom_id_pairs
            ])
            one   = ['ball/ball_geom', 'sawyer/allegro_right//unnamed_geom_12']
            two   = ['ball/ball_geom', 'sawyer/allegro_right//unnamed_geom_23']
            three = ['ball/ball_geom', 'sawyer/allegro_right//unnamed_geom_34']
            four  = ['ball/ball_geom', 'sawyer/allegro_right//unnamed_geom_45']
            for idx, (name, label) in enumerate(zip([one, two, three, four], ["Finger 1", "Finger 2", "Finger 3", "Finger 4"])):
                if name in geoms:
                    print(f"{label} in contact")
                    contact_fingers[idx] = True
                else:
                    print(f"{label} not in contact")
            if all(contact_fingers) and env.physics.data.ptr.contact.frame.shape[0] >= 4:
                print("All fingers in contact with the object.")
                in_contact = True

        # Evaluate the objective function and check its gradient
        fval = joint_space_objective(env, q_h, fingertip_names, in_contact, beta=beta)

        try:
            grad = numeric_gradient(
                joint_space_objective,
                q_h,
                env,
                fingertip_names,
                in_contact,
                beta=beta,
                eps=1e-3,
            )
        except TypeError:
            # Backward-compatible fallback if an old utils module is still cached.
            grad = numeric_gradient(
                joint_space_objective,
                q_h,
                env,
                fingertip_names,
                in_contact,
                1e-3,
            )
        grad_norm = np.linalg.norm(grad)
        print(f"Iter {it}, grad norm: {grad_norm:.6f}")
        min_norm = 1e-8
        step = grad / (grad_norm + min_norm)
        q_h_new = q_h.copy() - lr * step

        # Clip joint configuration to be in bounds
        qpos_new = env.physics.data.qpos.copy()
        qpos_new[env.q_h_slice] = q_h_new
        qpos_new = clip_to_valid_state(env.physics, qpos_new)
        q_h_new = qpos_new[env.q_h_slice].copy()

        # Evaluate the objective function with the new joint configuration to measure improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact, beta=beta)

        # Check if all fingertips are close to the object
        finger_positions = env.get_body_positions(fingertip_names)
        dists = [abs(env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)) for p in finger_positions]
        print(f"Iter {it}, fingertip distances: {dists}")
        for idx, d in enumerate(dists):
            print(f"Finger {idx+1} distance: {d:.5f} m, contact: {contact_fingers[idx]}")

        if in_contact:
            q_plus_last, q_minus_last = evaluate_force_closure_metrics(env)
            print(f"Iter {it}, force-closure metrics: Q+={q_plus_last:.6e}, Q-={q_minus_last:.6e}")
        else:
            q_plus_last, q_minus_last = float("inf"), float("inf")

        # Only update q_h if the objective function has improved
        improvement = fval - fval_new
        if fval_new < fval:
            q_h = q_h_new
            if return_history:
                history.append(q_h.copy())
            print(f"Iter {it}, objective={fval_new:.4f}, improvement={improvement:.4f}")
        else:
            # If no improvement, only reduce lr if all fingers are in contact
            if in_contact:
                lr = max(lr * 0.5, min_lr)
                print(f"Iter {it}, no improvement, reduce lr to {lr}")
            else:
                print(f"Iter {it}, no improvement, but not all fingers in contact. Keeping lr at {lr}")

        # Adaptive penalty: if not all fingers in contact, increase beta
        if not in_contact:
            beta = min(beta * 1.2, 1000)
            print(f"Increasing surface penalty beta to {beta}")
        else:
            beta = 50  # reset to default when in contact

        # Stopping conditions:
        # Stop when contact is established and force-closure metrics are optimized.
        if (
            in_contact
            and np.isfinite(q_plus_last)
            and np.isfinite(q_minus_last)
            and q_plus_last <= 1e-5
            and q_minus_last <= 0.0
        ):
            print(
                f"Stopping: all fingers in contact with optimized force closure, "
                f"Q+={q_plus_last:.6e}, Q-={q_minus_last:.6e}."
            )
            break

        # Only stop for very small improvement when force-closure criteria are already met.
        if (
            improvement < 1e-6
            and in_contact
            and np.isfinite(q_plus_last)
            and np.isfinite(q_minus_last)
            and q_plus_last <= 1e-5
            and q_minus_last <= 0.0
        ):
            print(
                f"Stopping: tiny improvement with force closure satisfied "
                f"(Q+={q_plus_last:.6e}, Q-={q_minus_last:.6e})."
            )
            break

        # If no improvement but not all fingers are in contact, keep optimizing
        if improvement < 1e-6 and not in_contact:
            print("No improvement but not all fingers in contact. Increasing beta and continuing.")
            # Optionally: reset lr, increase beta, or just continue
            lr = initial_lr
            beta = min(beta * 1.5, 2000)
            continue
    if return_history:
        return q_h, history
    return q_h


def evaluate_force_closure_metrics(env: AllegroHandEnv.AllegroHandEnv,
                                   friction_coeff=0.5,
                                   num_friction_cone_approx=4):
    contact_frames, contact_positions = env.get_contact_normals_and_positions(env.physics.data.ptr.contact)
    if len(contact_frames) == 0 or len(contact_positions) == 0:
        return float("inf"), float("inf")

    directions_list = []
    for i in range(len(contact_frames)):
        directions_i = build_friction_cone(contact_frames[i], friction_coeff, num_friction_cone_approx)
        directions_list.append(directions_i)

    G = build_grasp_matrix(contact_positions, directions_list, origin=env.sphere_center)
    if G.size == 0 or G.ndim != 2 or not np.all(np.isfinite(G)):
        return float("inf"), float("inf")

    q_plus = optimize_necessary_condition(G)
    q_minus = optimize_sufficient_condition(G)
    return q_plus, q_minus


def joint_space_objective(env: AllegroHandEnv.AllegroHandEnv,
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=100, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4,
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
        if G.size == 0 or G.ndim != 2 or not np.all(np.isfinite(G)):
            return 1e6 + beta * surface_penalty

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
    normal = np.asarray(normal, dtype=float).reshape(-1)
    if normal.size < 9 or not np.all(np.isfinite(normal)):
        return [np.array([0.0, 0.0, 1.0]) for _ in range(num_approx)]

    n = normal[0:3]
    tangent_1 = normal[3:6]
    tangent_2 = normal[6:9]
    
    d_theta = 2 * np.pi / num_approx
    
    friction_cones_vecs = []
    for i in range(num_approx):
        vec = mu * (np.cos(d_theta * i) * tangent_1 + np.sin(d_theta * i) * tangent_2) + n
        if not np.all(np.isfinite(vec)):
            vec = np.array([0.0, 0.0, 1.0])
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
        pos_vec = np.asarray(positions[i_, :], dtype=float)   # (3,)
        if pos_vec.size != 3 or not np.all(np.isfinite(pos_vec)):
            continue
        for j_ in range(len(friction_cones[i_])):
            dir_vec = np.asarray(friction_cones[i_][j_], dtype=float) # (3,)
            if dir_vec.size != 3 or not np.all(np.isfinite(dir_vec)):
                continue
            G_ji = np.hstack((dir_vec, np.cross((pos_vec - origin), dir_vec))) # (6,)
            if not np.all(np.isfinite(G_ji)):
                continue
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
    # Change G to float type
    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.size == 0 or G.shape[1] == 0 or not np.all(np.isfinite(G)):
        # if G has no instances or the grasp matrix is not a 2D matrix
        return 1e6

    # N = number of adjoint grasps
    N = G.shape[1]
    # wrench_dim = per adjoint dimension (should be 6 for 3D)
    wrench_dim = G.shape[0]
    if wrench_dim == 0:
        return 1e6

    def obj(alpha):
        w = G @ alpha
        return float(np.dot(w, w))

    x0 = np.full(N, 1.0 / N)
    bounds = [(0.0, None)] * N
    constraints = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]

    try:
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )
    except Exception:
        return 1e6

    if not res.success or not np.all(np.isfinite(res.x)):
        return 1e6

    val = float(np.linalg.norm(G @ res.x))
    return val if np.isfinite(val) else 1e6
    
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
    if G.ndim != 2 or G.size == 0 or G.shape[1] == 0 or not np.all(np.isfinite(G)):
        return 1e6

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

        A_eq = np.zeros((wrench_dim + 1, N + 1), dtype=float)
        A_eq[:wrench_dim, :N] = G
        A_eq[:wrench_dim, -1] = -qk
        A_eq[wrench_dim, :N] = 1.0

        b_eq = np.zeros(wrench_dim + 1)
        b_eq[wrench_dim] = 1.0

        if not np.all(np.isfinite(A_eq)) or not np.all(np.isfinite(b_eq)):
            continue

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            d_q_vals.append(float(res.x[-1]))

    if not d_q_vals:
        return 1e6

    out = -min(d_q_vals)
    return out if np.isfinite(out) else 1e6
