import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
from utils import *
import grasp_synthesis

import casadi as ca

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""


def fingertip_surface_distances(env: AllegroHandEnv.AllegroHandEnv,
                                q_h: np.array,
                                fingertip_names: list[str]) -> np.array:
    """
    Signed surface-to-surface distances between the fingertip rubber spheres and the ball.
    Negative means penetration. The fingertip names correspond to helper bodies whose
    collision geoms are spheres of radius 0.012 in the notebook setup.
    """
    env.set_configuration(q_h)
    finger_positions = env.get_body_positions(fingertip_names)
    fingertip_radius = 0.012
    return np.array([
        np.linalg.norm(p - env.sphere_center) - (env.sphere_radius + fingertip_radius)
        for p in finger_positions
    ], dtype=float)


def compute_in_contact(env: AllegroHandEnv.AllegroHandEnv,
                       q_h: np.array,
                       fingertip_names: list[str],
                       distance_threshold: float = 5e-3,
                       min_fingers_in_contact: int = 3) -> tuple[bool, np.array]:
    """
    Distance-threshold contact check used consistently across synthesize_grasp and
    joint_space_objective. A fingertip counts as 'in contact' when it is close to the
    sphere surface (or slightly penetrating).
    """
    dists = fingertip_surface_distances(env, q_h, fingertip_names)
    near_surface = dists <= distance_threshold
    return bool(np.sum(near_surface) >= min_fingers_in_contact), dists

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

    def min_surface_distance(q_h_local: np.array) -> float:
        """Minimum signed distance to sphere surface across fingertips (negative => penetration)."""
        dists = fingertip_surface_distances(env, q_h_local, fingertip_names)
        return float(np.min(dists)) if len(dists) else float("inf")

    def surface_distances(q_h_local: np.array) -> np.array:
        """Signed distance to sphere surface for each fingertip body."""
        return fingertip_surface_distances(env, q_h_local, fingertip_names)

    penetration_tol = 1e-4
    contact_achieved = False
    # Persistent per-finger contact state with hysteresis.
    enter_contact_thresh = 5e-3
    exit_contact_thresh = 1e-2
    latched_contacted_fingers = np.zeros(4, dtype=bool)
    for it in range(max_iters):
        in_contact, finger_d_before = compute_in_contact(
            env,
            q_h,
            fingertip_names,
            distance_threshold=enter_contact_thresh,
            min_fingers_in_contact=3,
        )
        entering_now = finger_d_before <= enter_contact_thresh
        staying_in_contact = latched_contacted_fingers & (finger_d_before <= exit_contact_thresh)
        latched_contacted_fingers = entering_now | staying_in_contact
        if in_contact and not contact_achieved:
            lr *= 0.04
            contact_achieved = True
            print("FINGERTIPS ARE CLOSE ENOUGH TO BALL; entering refinement mode")
        else:
            print("contacted_fingers:", latched_contacted_fingers.astype(int), "distances:", np.round(finger_d_before, 4))
        
        # Evaluate the objective function and check its gradient
        fval = joint_space_objective(env, q_h, fingertip_names, in_contact)
        grad = numeric_gradient(joint_space_objective, q_h, env, fingertip_names, in_contact)

        # # Zero gradient for any finger that has already made its contact, and keep its joints fixed.
        # for fi in range(4):
        #     if frozen_finger[fi]:
        #         grad[fi * 4 : (fi + 1) * 4] = 0.0

        # Update the joint configuration
        min_d_before = min_surface_distance(q_h)
        q_h_new = q_h.copy() - lr*grad

        # # Hard-freeze joints for fingers already in contact.
        # for fi in range(4):
        #     if frozen_finger[fi]:
        #         q_h_new[fi * 4 : (fi + 1) * 4] = q_h[fi * 4 : (fi + 1) * 4]
        
        # Clip joint configuration to be in bounds
        qpos_new = env.physics.data.qpos.copy()
        qpos_new[env.q_h_slice] = q_h_new
        qpos_new = clip_to_valid_state(env.physics, qpos_new)
        q_h_new = qpos_new[env.q_h_slice].copy()

        # Soft post-contact lock: once a fingertip is near the ball, do not allow
        # that finger to move further inward. Revert only that finger's 4 joints.
        finger_d_after = surface_distances(q_h_new)
        for fi in range(4):
            if latched_contacted_fingers[fi] and (
                #(finger_d_after[fi] < 0.0) or # NO TOLERANCE FOR NEGATIVE DRIFT AKA COLLISIONS.
                (finger_d_after[fi] < finger_d_before[fi] - penetration_tol) or
                (finger_d_after[fi] > exit_contact_thresh)
            ):
                q_h_new[fi * 4 : (fi + 1) * 4] = q_h[fi * 4 : (fi + 1) * 4]

        # Recompute distances after any per-finger revert.
        finger_d_after = surface_distances(q_h_new)

        # Reject if penetration got worse (more negative signed distance).
        min_d_after = min_surface_distance(q_h_new)

        if (min_d_after < min_d_before - penetration_tol) and (min_d_after < 0.0):
            lr *= 0.5
            print(
                f"Iter {it}, reject step (penetration worsened: {min_d_before:.5f} -> {min_d_after:.5f}), lr -> {lr}"
            )
            env.set_configuration(q_h)
            if lr < 1e-6:
                print("penetration worsened and lr is too small, RETURNING")
                break
            continue

        in_contact_new, _ = compute_in_contact(env, q_h_new, fingertip_names)


        # Evaluate the objective function with the new joint configuration to measure improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact_new, penetration_weight=20000)

        # Only update q_h if the objective function has improved
        if fval_new < fval:
            q_h = q_h_new
            if return_history:
                history.append(q_h.copy())
            improvement = fval - fval_new 
            print(f"Iter {it}, objective={fval_new:.4f}, improvement={improvement:.4f}")
            if contact_achieved and improvement < 1e-6:
                print("improvement is too small, RETURNING")
                break
        else:
            # If no improvement, reduce lr or break
            lr *= 0.5
            print(f"Iter {it}, no improvement, reduce lr to {lr}")
            if lr < 1e-6:
                print("lr is too small, RETURNING")
                break
    print("RETURNING")
    if return_history:
        return q_h, history
    return q_h

def joint_space_objective(env: AllegroHandEnv.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10, 
                          penetration_weight=600,
                          tweak_surf_penalty_at_contact=True,
                          tweak_surf_penalty_at_contact_value=0.04,
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
    for p in finger_positions:
        d = env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)
        # Penalize penetration more heavily so early-contact fingers do not clip through the sphere.
        if d < 0:
            surface_penalty += penetration_weight * d * d
            outside_distances.append(0.0)
        else:
            surface_penalty += d * d

            # Use outside distance for synchronization so all fingertips close in together.
            outside_distances.append(d)

    outside_distances = np.array(outside_distances)
    sync_penalty = np.var(outside_distances)
    surface_penalty += sync_weight * sync_penalty

    # Inter-finger collision penalty
    collision_total_penalty = 0.0
    collision_penalty = 0.0
    ## THIS IS ONLY THHING CHANGED from working version,, min_finger_dist was 0.025
    min_finger_dist = 0.04  # meters
    collision_weight = 500.0
    n_fingers = len(finger_positions)
    for i in range(n_fingers):
        for j in range(i+1, n_fingers):
            dist = np.linalg.norm(finger_positions[i] - finger_positions[j])
            if dist < min_finger_dist:
                print(f"Collision: Fingers {i+1} and {j+1} too close (d={dist:.4f})")
                collision_penalty += (min_finger_dist - dist) ** 2
    collision_total_penalty = collision_weight * collision_penalty

    in_house_in_contact, _ = compute_in_contact(env, q_h, fingertip_names)

    if not in_house_in_contact:
        print("not in contact or less than 4 contact frames")
        return beta * (surface_penalty) + collision_total_penalty
    else: # Fingers are in contact, so we calculate Q+ and Q- penalty
        print("in contact and more than 4 contact frames")
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
            print("Use necessary condition")
            score_fc = Q_plus_dist
        else:
            Q_minus_dist = optimize_sufficient_condition(G)
            print("Use sufficient condition")
            score_fc = Q_minus_dist
        final_coll_penalty_values = tweak_surf_penalty_at_contact_value *(beta * (surface_penalty) + collision_total_penalty)
        print("final_coll_penalty_values are these :", final_coll_penalty_values)
        return score_fc + final_coll_penalty_values
    
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

def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv):
    """
    Returns the result of the L2 optimization on G (Q+ distance)

    Parameters
    ----------
    G: grasp matrix
    env: AllegroHandEnv instance (can use to access physics)
    """
    #YOUR CODE HERE

    # extract variable dimensions
    N_ = G.shape[1] # number of force magitudes, one for each cone direction

    # construct a casadi optimization problem
    opti = ca.Opti()
    # define variables and parameters
    X = opti.variable(N_, 1)    # (N= num_appro * num_contact,)
    # set initial and constraints
    opti.set_initial(X[:], 0.0)
    # eps_ = 1e-7
    eps_ = 0.0
    opti.subject_to(X[:] >= eps_)
    opti.subject_to(ca.sum1(X) == 1)
    
    # define objective function
    J_cost = (X.T @ G.T) @ (G @ X)
    opti.minimize(J_cost)
    
    # set solver
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"max_iter": 3000, "print_level": 1, "sb": "yes"}
    )

    # solve the optimization problem
    sol = opti.solve()
    
    # obtain the optimization output
    alpha_vec = np.array(sol.value(X[:])).flatten().reshape(N_,1)

    return np.sqrt( (alpha_vec.T @ G.T) @ (G @ alpha_vec) ).squeeze()
    # return ( (alpha_vec.T @ G.T) @ (G @ alpha_vec)).squeeze()



def optimize_sufficient_condition(G: np.array, M=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    M: number of approximations to the norm ball

    Returns the Q- distance
    """
    # extract dimensions
    D_, N_ = G.shape        # dimension of workspace and number of friction cone boundary vectors
    q_M_arr = _generate_sphere_samples(dim=D_, num_pts=M).T  # (D_, M)

    ## 1. construct a casadi optimization problem
    opti = ca.Opti()

    ## 2. define variables and parameters
    # paramters that remain constant during optimization
    G_mat = opti.parameter(D_, N_)  # grasp matrix
    Q_mat = opti.parameter(D_, M)   # directions matrix
    opti.set_value(G_mat, G)        
    opti.set_value(Q_mat, q_M_arr)
    # decision variables to be changed during the optimization
    r = opti.variable()             # ball radius
    A = opti.variable(N_, M)        # force matrix, each col. is a force vector for one direction

    ## 3. set initial and constraints
    # initial values for decision variables
    eps_ = 0.0 # [MT]: may consider a value >0 for numerical stability
    opti.set_initial(r, eps_)
    opti.set_initial(A, eps_)
    # constraints
    opti.subject_to(r >= eps_)
    opti.subject_to(ca.vec(A) >= eps_)
    # apply constraints to each direction in the sphere
    for k in range(M):
        opti.subject_to(ca.sum1(A[:, k]) == 1)
        opti.subject_to(G_mat @ A[:, k] == Q_mat[:, k] * r)

    ## 4. define objective function
    # maximize common radius, 
    # referring to the max ball radius around origin within discretized friction cone
    opti.minimize(-r)

    ## 5 set solver and solve the optimization problem
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"max_iter": 3000, "print_level": 1}
    )

    # optimization problem may not be always well constructed, using try-except
    try:
        sol = opti.solve()
        return -float(sol.value(r)) # notice the negative sign
    except Exception as err:
        if 1:
            print(err)
    
    return 0.0  # return 0.0 radius if no valid solution
