import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
from utils import *

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""


def _safe_geom_name(model_ptr, geom_id: int):
    name = mj.mj_id2name(model_ptr, mj.mjtObj.mjOBJ_GEOM, int(geom_id))
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return "" if name is None else str(name)


def _collect_object_fingertip_contacts(env: AllegroHandEnv.AllegroHandEnv, fingertip_names: list[str]):
    """Collect object-fingertip contacts with frame/position metadata."""
    physics = env.physics
    model_ptr = physics.model.ptr
    contact_struct = physics.data.ptr.contact

    object_geom_id = int(mj.mj_name2id(model_ptr, mj.mjtObj.mjOBJ_GEOM, env.object_name))
    if object_geom_id < 0:
        return []

    fingertip_body_ids = set()
    for body_name in fingertip_names:
        try:
            fingertip_body_ids.add(int(physics.model.body(body_name).id))
        except Exception:
            continue

    if not fingertip_body_ids:
        return []

    geom_bodyid = np.asarray(model_ptr.geom_bodyid, dtype=int)
    fingertip_geom_ids = set(np.flatnonzero(np.isin(geom_bodyid, list(fingertip_body_ids))).tolist())
    if not fingertip_geom_ids:
        return []

    contacts = []
    for i in range(int(physics.data.ncon)):
        pair = contact_struct.geom[i]
        g0, g1 = int(pair[0]), int(pair[1])

        if g0 == object_geom_id and g1 in fingertip_geom_ids:
            object_geom, finger_geom = g0, g1
        elif g1 == object_geom_id and g0 in fingertip_geom_ids:
            object_geom, finger_geom = g1, g0
        else:
            continue

        frame = np.asarray(contact_struct.frame[i], dtype=float).reshape(-1)
        pos = np.asarray(contact_struct.pos[i], dtype=float).reshape(-1)
        if frame.size < 9 or pos.size < 3:
            continue

        normal = frame[0:3]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12 or not np.all(np.isfinite(normal)):
            continue
        normal = normal / normal_norm

        finger_body_id = int(geom_bodyid[finger_geom])
        contacts.append(
            {
                "index": i,
                "object_geom_id": object_geom,
                "finger_geom_id": finger_geom,
                "finger_body_id": finger_body_id,
                "normal": normal,
                "frame": frame,
                "pos": pos[0:3],
                "geom_names": (
                    _safe_geom_name(model_ptr, object_geom),
                    _safe_geom_name(model_ptr, finger_geom),
                ),
            }
        )

    return contacts


def _tangential_slip_penalty(env: AllegroHandEnv.AllegroHandEnv, contacts: list[dict]):
    """Penalize relative tangential velocity to reduce fingertip sliding."""
    if not contacts:
        return 0.0

    data = env.physics.data
    geom_bodyid = np.asarray(env.physics.model.ptr.geom_bodyid, dtype=int)

    slip = 0.0
    valid = 0
    for c in contacts:
        object_body = int(geom_bodyid[c["object_geom_id"]])
        finger_body = int(geom_bodyid[c["finger_geom_id"]])

        v_obj = np.asarray(data.cvel[object_body][3:6], dtype=float)
        v_finger = np.asarray(data.cvel[finger_body][3:6], dtype=float)
        if v_obj.size != 3 or v_finger.size != 3:
            continue
        if not np.all(np.isfinite(v_obj)) or not np.all(np.isfinite(v_finger)):
            continue

        normal = np.asarray(c["normal"], dtype=float)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm

        v_rel = v_finger - v_obj
        v_tan = v_rel - np.dot(v_rel, normal) * normal
        slip += float(np.dot(v_tan, v_tan))
        valid += 1

    if valid == 0:
        return 0.0
    return slip / valid


def _grasp_status(env: AllegroHandEnv.AllegroHandEnv, q_h: np.array, fingertip_names: list[str]):
    env.set_configuration(q_h)
    finger_positions = env.get_body_positions(fingertip_names)
    signed_distances = np.array(
        [env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius) for p in finger_positions],
        dtype=float,
    )
    contacts = _collect_object_fingertip_contacts(env, fingertip_names)
    touched_ids = {c["finger_body_id"] for c in contacts}
    return {
        "signed_distances": signed_distances,
        "max_abs_distance": float(np.max(np.abs(signed_distances))) if signed_distances.size else np.inf,
        "num_contacts": len(contacts),
        "num_fingers_touching": len(touched_ids),
    }

def synthesize_grasp(env: AllegroHandEnv.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=2000, 
                         lr=4,
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
        # Evaluate objective at current state; objective computes contact status internally.
        fval = joint_space_objective(env, q_h, fingertip_names, False)
        grad = numeric_gradient(joint_space_objective, q_h, env, fingertip_names, False)

        # Update the joint configuration
        q_h_new = q_h.copy() - lr*grad
        
        # Clip joint configuration to be in bounds
        qpos_new = env.physics.data.qpos.copy()
        qpos_new[env.q_h_slice] = q_h_new
        qpos_new = clip_to_valid_state(env.physics, qpos_new)
        q_h_new = qpos_new[env.q_h_slice].copy()

        # Evaluate the objective function with the new joint configuration to measure improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, False)

        # Only update q_h if the objective function has improved
        if fval_new < fval:
            q_h = q_h_new
            if return_history:
                history.append(q_h.copy())
            improvement = fval - fval_new 
            status = _grasp_status(env, q_h, fingertip_names)
            print(
                f"Iter {it}, objective={fval_new:.4f}, improvement={improvement:.4f}, "
                f"contacts={status['num_contacts']}, touching={status['num_fingers_touching']}, "
                f"max|dist|={status['max_abs_distance']:.4f}"
            )
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
                          beta=80, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4,
                          eps=0.00001,
                          distance_tol=0.01,
                          q_plus_switch=5e-3,
                          slip_weight=10.0,
                          balance_weight=20.0):
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
    signed_distances = np.array(
        [env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius) for p in finger_positions],
        dtype=float,
    )
    abs_distances = np.abs(signed_distances)
    surface_penalty = float(np.sum(signed_distances * signed_distances))

    # Encourage all fingers to approach the surface together.
    distance_balance_penalty = float(np.var(abs_distances))

    contacts = _collect_object_fingertip_contacts(env, fingertip_names)
    touched_fingers = {c["finger_body_id"] for c in contacts}
    all_fingers_near_surface = bool(abs_distances.size == len(fingertip_names) and np.all(abs_distances <= distance_tol))
    strict_ready_for_q_minus = (
        len(contacts) >= len(fingertip_names)
        and len(touched_fingers) >= len(fingertip_names)
        and all_fingers_near_surface
    )

    # Before meaningful contact, drive fingertips toward the object and avoid stalling non-thumb digits.
    if len(contacts) == 0:
        return beta * surface_penalty + balance_weight * distance_balance_penalty

    contact_positions = np.array([c["pos"] for c in contacts], dtype=float)
    directions_list = [
        build_friction_cone(c["frame"], friction_coeff, num_friction_cone_approx) for c in contacts
    ]
    G = build_grasp_matrix(contact_positions, directions_list, origin=env.sphere_center)

    if G.size == 0 or G.shape[0] != 6 or G.shape[1] < 6 or np.linalg.matrix_rank(G) < 4:
        return beta * surface_penalty + balance_weight * distance_balance_penalty

    Q_plus_dist = optimize_necessary_condition(G, env)
    if not np.isfinite(Q_plus_dist):
        Q_plus_dist = 1e6

    score_fc = Q_plus_dist
    if strict_ready_for_q_minus and Q_plus_dist <= q_plus_switch:
        Q_minus_dist = optimize_sufficient_condition(G)
        if np.isfinite(Q_minus_dist):
            score_fc = min(Q_plus_dist, Q_minus_dist)

    contact_progress = min(1.0, len(touched_fingers) / max(1, len(fingertip_names)))
    fc_weight = 0.25 + 0.75 * contact_progress
    slip_penalty = _tangential_slip_penalty(env, contacts)

    return (
        beta * surface_penalty
        + balance_weight * distance_balance_penalty
        + slip_weight * slip_penalty
        + fc_weight * score_fc
    )
    
    
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
        return [np.array([1.0, 0.0, 0.0]) for _ in range(num_approx)]

    n = normal[0:3]
    tangent_1 = normal[3:6]
    tangent_2 = normal[6:9]
    
    d_theta = 2 * np.pi / num_approx
    
    friction_cones_vecs = []
    for i in range(num_approx):
        vec = mu * (np.cos(d_theta * i) * tangent_1 + np.sin(d_theta * i) * tangent_2) + n
        if not np.all(np.isfinite(vec)):
            vec = np.array([1.0, 0.0, 0.0])
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-12:
            vec = np.array([1.0, 0.0, 0.0])
        else:
            vec = vec / norm_vec
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

    Returns the Q- distance
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
        if res.success and np.all(np.isfinite(res.x)):
            d_q_vals.append(float(res.x[-1]))

    if not d_q_vals:
        return 1e6

    out = -min(d_q_vals)
    return out if np.isfinite(out) else 1e6
