import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
import grasp_synthesis
import types
from utils import *

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: grasp_synthesis.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=1000, 
                         lr=0.1):
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
    
    # Require: qh, max iter, λ
    #     iter ← 0
    #     qh ← init configuration
    #     in contact = False
    #     while iter < max iter do
        #     if all fingers touching then
        #       made contact ← True
        #     end if
        #     f ← joint space objective
        #     qnew
        #     h = qh − λ∇qh f (qh)
        #     improvement ← f (qh) − f (qnew
        #     h )
        #     if improvement > 0 then
            #     qh ← qnew
            #     h
            #     if improvement < 1e−6 then
            #       break
            #     end if
        #     end if
        #     end while
    #     return qh
    # Input validation
    if q_h_init is None or len(q_h_init) == 0:
        raise ValueError("q_h_init cannot be empty")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    
    q_h = q_h_init.copy()
    in_contact = False
    
    for iter in range(max_iters):
        
        # Check if all four fingertips are in contact with the object
        # Use the environment's contact detection which properly filters for ball contacts
        if env.physics.data.ncon >= 4:
            contact_normals, contact_positions = env.get_contact_normals_and_positions(
                env.physics.data.contact
            )
            # If we have 4+ valid contacts with the ball, we have contact
            if len(contact_positions) >= 4:
                in_contact = True
        
        # Evaluate the objective function
        f = joint_space_objective(env, q_h, fingertip_names, in_contact)
        
        # Update the joint configuration
        grad_f = numeric_gradient(joint_space_objective, q_h, env, fingertip_names, in_contact)
        q_h_new = q_h.copy() - lr*grad_f 

        # Clip in full qpos space so MuJoCo joint indices/ranges are applied correctly.
        qpos_full = env.physics.data.qpos.copy()
        qpos_full[env.q_h_slice] = q_h_new
        qpos_full = clip_to_valid_state(env.physics, qpos_full)
        q_h_new = qpos_full[env.q_h_slice].copy()

        # Evaluate the objective function with the new joint configuration to measure improvement
        f_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact)

        # Only update q_h if the objective function has improved
        if f_new < f:
            q_h = q_h_new
            improvement = f - f_new 
            print(f"Iter {iter}, objective={f_new:.4f}, improvement={improvement:.4f}")
            if improvement < 1e-6:
                break
        else:
            # If no improvement, reduce lr or break
            lr *= 0.5
            print(f"Iter {iter}, no improvement, reduce lr to {lr}")
            if lr < 1e-6:
                break
    return q_h

def joint_space_objective(env: grasp_synthesis.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4,
                          eps=0.000001):
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
    # Require: qh, β, fingertip positions, in contact, Q plus thresh
    # D = get total dist from sphere(positions)
    # if not in contact then return βD
    # else
        # normals ← get contact normals()
        # F C ← build discrete friction cone(normals)
        # G ← build grasp map(FC)
        # fc loss ← optimize necessary condition(G)
        # if fc loss < Q plus thresh then
        #   fc loss ← optimize sufficient condition(G)
        # end if
        # return fc loss + βD
    # end if


    env.set_configuration(q_h)
    finger_positions = env.get_body_positions(fingertip_names)

    # Penalty for distance from surface
    surface_penalty = 0.0
    for p in finger_positions:
        d = env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)
        surface_penalty += d*d
    if not in_contact:
        return beta * surface_penalty
    else: # Fingers are in contact, so we calculate Q+ and Q- penalty
        # Create friction cone
        try:
            contact_frames, contact_positions = env.get_contact_normals_and_positions(env.physics.data.contact)
        except (IndexError, ValueError) as e:
            print(f"Error getting contact information: {e}")
            return float('inf')
        
        if len(contact_frames) == 0:
            return beta * surface_penalty
        
        directions_list = []
        for i in range(len(contact_frames)):
            contact_frame = contact_frames[i]
            directions_i = build_friction_cone(contact_frame, friction_coeff, num_friction_cone_approx)
            directions_list.append(directions_i)

        try:
            G = build_grasp_matrix(contact_positions, directions_list, origin=env.sphere_center)
        except (ValueError, IndexError) as e:
            print(f"Error building grasp matrix: {e}")
            return float('inf')

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
    if normal.shape[0] < 9:
        raise ValueError(f"normal must have at least 9 elements, got {normal.shape[0]}")
    if mu < 0:
        raise ValueError("Friction coefficient mu must be non-negative")
    if num_approx < 1:
        raise ValueError("num_approx must be at least 1")
    
    n = normal[0:3]
    tangent_1 = normal[3:6]
    tangent_2 = normal[6:9]
    
    friction_cones_vecs = []
    for angle in np.linspace(0, 2*np.pi, num_approx, endpoint=False):
        # Tangential component
        tangent_f = np.cos(angle) * tangent_1 + np.sin(angle) * tangent_2
        # Friction cone force: normal + scaled tangent
        f = n + mu * tangent_f
        norm_f = np.linalg.norm(f)
        if norm_f > 0:
            friction_cones_vecs.append(f / norm_f)  # normalize
        else:
            friction_cones_vecs.append(f)
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
    if len(positions) == 0 or len(friction_cones) == 0:
        raise ValueError("positions and friction_cones cannot be empty")
    if len(positions) != len(friction_cones):
        raise ValueError(f"Number of positions ({len(positions)}) must match number of friction cones ({len(friction_cones)})")
    
    G_list = []
    for i, contact_point in enumerate(positions):
        r = contact_point - origin
        
        for cone_direction_f in friction_cones[i]:
            torque = np.cross(r, cone_direction_f)
            wrench = np.hstack([cone_direction_f, torque])
            G_list.append(wrench)
    
    if len(G_list) == 0:
        raise ValueError("No wrenches generated from friction cones")
    
    G = np.column_stack(G_list)
    return G
            

def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv):
    """
    Returns the result of the L2 optimization on G (Q+ distance)

    Parameters
    ----------
    G: grasp matrix
    env: AllegroHandEnv instance (can use to access physics)
    """
    # Q+ distance: min ||G alpha||_2 subject to alpha in simplex.
    n_dirs = G.shape[1]

    def f_objective(alpha):
        return np.linalg.norm(G @ alpha)

    x0 = np.ones(n_dirs) / n_dirs
    constraints = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0},)
    bounds = [(0.0, None)] * n_dirs

    result = minimize(
        f_objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        return float('inf')
    return float(result.fun)

def optimize_sufficient_condition(G: np.array, M=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    M: number of approximations to the norm ball

    Returns the Q- distance. Negative values indicate force-closure sufficiency.
    """

    n_dirs = G.shape[1]

    # Variables are x = [alpha_1 ... alpha_N, r].
    # We solve d_Q^-(k) = min -r, s.t. G alpha = r q_k, sum(alpha)=1, alpha>=0, r>=0.
    
    c = np.zeros(n_dirs + 1) #  the objective function c' * x = [0 * alpha_1 ... 0 * alpha_N, -r] = [0 ... 0, -r]
    c[-1] = -1.0 # -r
    bounds = [(0.0, None)] * (n_dirs + 1) # alpha_1 ... alpha_N, r all needs to be non-negative

    # generate M = 20 random unit vectors to present the unit ball
    rng = np.random.default_rng(0)
    rand_directions = rng.normal(size=(M, 6))
    norms = np.linalg.norm(rand_directions, axis=1, keepdims=True)
    # safety precaution: prevents division by 0, in case the rng generates a 0 vector
    norms[norms == 0.0] = 1.0
    rand_directions = rand_directions / norms

    d_k_values = []
    for q_k in rand_directions:
        A_eq = np.zeros((7, n_dirs + 1))
        A_eq[:6, :n_dirs] = G
        A_eq[:6, -1] = -q_k
        A_eq[6, :n_dirs] = 1.0
        b_eq = np.zeros(7)
        b_eq[6] = 1.0

        result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            d_k_values.append(float(result.fun))
        else:
            # Infeasible direction => no positive interior radius along this ray.
            d_k_values.append(0.0)

    if not d_k_values or len(d_k_values) == 0:
        return 0.0  # Return 0 instead of inf (no force closure)
    return float(np.max(d_k_values))
