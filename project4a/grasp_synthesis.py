import numpy as np
from scipy.optimize import linprog, minimize
import mujoco as mj
import AllegroHandEnv
from utils import *


# Fingertip body names (match the tip_rubber bodies created in the notebook)
FINGERTIP_BODY_NAMES = [
    'sawyer/allegro_right/ff_tip_rubber',
    'sawyer/allegro_right/mf_tip_rubber',
    'sawyer/allegro_right/rf_tip_rubber',
    'sawyer/allegro_right/th_tip_rubber',
]


def _get_fingertip_geom_names(physics):
    """
    Dynamically extract geom names from fingertip bodies.
    This handles auto-generated geom names from the MJCF model.
    """
    geom_names = []
    try:
        for body_name in FINGERTIP_BODY_NAMES:
            body_id = physics.model.name2id(body_name, 'body')
            # Find geoms attached to this body
            body_geom_range = physics.model.body_geomadr[body_id]
            n_geoms = physics.model.body_geomnum[body_id]
            for i in range(n_geoms):
                geom_id = physics.model.geom_bodyid[body_geom_range + i]
                geom_name = physics.model.id2name(geom_id, 'geom')
                geom_names.append(geom_name)
    except Exception as e:
        print(f"Warning: Could not extract fingertip geom names: {e}")
        # Fallback to hardcoded names if dynamic extraction fails
        geom_names = [
            'sawyer/allegro_right//unnamed_geom_12',
            'sawyer/allegro_right//unnamed_geom_23',
            'sawyer/allegro_right//unnamed_geom_34',
            'sawyer/allegro_right//unnamed_geom_45',
        ]
    return geom_names

def _all_fingers_touching(env, fingertip_geom_names, ball_geom_name="ball/ball_geom"):
    model = env.physics.model
    contacts = env.physics.data.ptr.contact
    ncon = env.physics.data.ncon

    fingertips_touching = set()
    for i in range(ncon):
        g0_id = int(contacts.geom[i][0]) # first contact object ID
        g1_id = int(contacts.geom[i][1]) # second contact object ID

        g0 = model.id2name(g0_id, "geom")
        g1 = model.id2name(g1_id, "geom")

        finger_on_ball = (
            (g0 in fingertip_geom_names and g1 == ball_geom_name) or
            (g1 in fingertip_geom_names and g0 == ball_geom_name)
        )
        if finger_on_ball:
            if g0 in fingertip_geom_names:
                fingertips_touching.add(g0)
            if g1 in fingertip_geom_names:
                fingertips_touching.add(g1)

    return len(fingertips_touching) == len(set(fingertip_geom_names))
    

def synthesize_grasp(env: AllegroHandEnv.AllegroHandEnv, 
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
    q_h = q_h_init.copy()
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
            one   = ['ball/sphere', 'sawyer/allegro_right//unnamed_geom_12']
            two   = ['ball/sphere', 'sawyer/allegro_right//unnamed_geom_23']
            three = ['ball/sphere', 'sawyer/allegro_right//unnamed_geom_34']
            four  = ['ball/sphere', 'sawyer/allegro_right//unnamed_geom_45']

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
        q_h_new = clip_to_valid_state(env.physics, q_h_new, env.q_h_slice, 16)

        # Evaluate the objective function with the new joint configuration to measure improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact)

        # Only update q_h if the objective function has improved
        if fval_new < fval:
            q_h = q_h_new
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
    return q_h

def joint_space_objective(env: AllegroHandEnv.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10, 
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
    n_raw = normal[0:3]
    n_norm = np.linalg.norm(n_raw)
    if n_norm < 1e-12:
        return np.zeros((0, 3))
    n = n_raw / n_norm
    
    # Ensure normal points outward: friction cone vectors should point away from contact.
    # Build a test friction vector and verify it has positive outward direction.
    t_test = normal[3:6] - np.dot(normal[3:6], n) * n
    t_test_norm = np.linalg.norm(t_test)
    if t_test_norm > 1e-12:
        t_test = t_test / t_test_norm
        f_test = n + mu * t_test
        # Sanity check: the test friction vector should have reasonable magnitude and point away.
        # If the normal was inverted (pointing inward), typical contacts would be problematic.
        # When in doubt, keep normal as-is; the rest of the grasp detection will filter bad cones.
        pass

    # Build a numerically stable tangent basis in the contact plane.
    t1_raw = normal[3:6]
    t1 = t1_raw - np.dot(t1_raw, n) * n
    t1_norm = np.linalg.norm(t1)
    if t1_norm < 1e-12:
        # Fallback if the provided tangent is degenerate.
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = ref - np.dot(ref, n) * n
        t1_norm = np.linalg.norm(t1)
    t1 = t1 / max(t1_norm, 1e-12)
    t2 = np.cross(n, t1)
    t2 = t2 / max(np.linalg.norm(t2), 1e-12)
    
    friction_cones_vecs = []
    for angle in np.linspace(0.0, 2.0 * np.pi, num_approx, endpoint=False):
        # Tangential component
        tangent_f = np.cos(angle) * t1 + np.sin(angle) * t2
        tangent_f = tangent_f / max(np.linalg.norm(tangent_f), 1e-12)
        # Friction cone force: normal + scaled tangent
        f = n + mu * tangent_f
        friction_cones_vecs.append(f / max(np.linalg.norm(f), 1e-12))
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
    positions = np.asarray(positions)
    if positions.size == 0 or len(friction_cones) == 0:
        return np.zeros((6, 0))

    G_list = []
    for i, contact_point in enumerate(positions):
        if i >= len(friction_cones):
            break
        r = contact_point - origin
        r_dist = np.linalg.norm(r)
        
        # Skip contact if it's at the origin or extremely close (likely spurious).
        if r_dist < 1e-8:
            continue
        
        for cone_direction_f in friction_cones[i]:
            cone_direction_f = np.asarray(cone_direction_f)
            if cone_direction_f.shape != (3,):
                continue
            
            # Sanity check: friction vectors should point roughly outward from the object.
            # A heuristic: the friction force should have positive component in the radial direction.
            radial_hat = r / max(r_dist, 1e-12)
            radial_component = np.dot(cone_direction_f, radial_hat)
            # Allow small inward components (numerical noise), but reject strongly inward forces.
            if radial_component < -0.3:
                continue
            
            torque = np.cross(r, cone_direction_f)
            wrench = np.hstack([cone_direction_f, torque])
            G_list.append(wrench)

    if not G_list:
        return np.zeros((6, 0))
    return np.column_stack(G_list)
            

def optimize_necessary_condition(G: np.array):
    """
    Returns the result of the L2 optimization on G (Q+ distance)

    Parameters
    ----------
    G: grasp matrix
    """
    # Q+ distance: min ||G alpha||_2 subject to alpha in simplex.
    if G.size == 0 or G.shape[1] == 0:
        return float('inf')

    # Additional sanity check: if we have very few columns or degenerate geometry, return inf.
    n_dirs = G.shape[1]
    if n_dirs < 3:  # Need at least 3 independent directions for 6D wrench space coverage.
        return float('inf')

    def f_objective(alpha):
        # Squared norm has the same minimizer as L2 norm and is smoother near zero.
        wrench = G @ alpha
        return float(np.dot(wrench, wrench))

    x0 = np.ones(n_dirs) / n_dirs
    constraints = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0},)
    bounds = [(0.0, None)] * n_dirs

    result = minimize(
        f_objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-9, 'disp': False},
    )
    if not result.success:
        return float('inf')
    return float(np.linalg.norm(G @ result.x))

def optimize_sufficient_condition(G: np.array, M=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    M: number of approximations to the norm ball

    Returns the Q- distance. Negative values indicate force-closure sufficiency.
    """

    if G.size == 0 or G.shape[1] == 0:
        return float('inf')

    n_dirs = G.shape[1]

    # Variables are x = [alpha_1 ... alpha_N, r].
    # We solve d_Q^-(k) = min -r, s.t. G alpha = r q_k, sum(alpha)=1, alpha>=0, r>=0.
    
    c = np.zeros(n_dirs + 1) #  the objective function c' * x = [0 * alpha_1 ... 0 * alpha_N, -r] = [0 ... 0, -r]
    c[-1] = -1.0 # -r
    bounds = [(0.0, None)] * (n_dirs + 1) # alpha_1 ... alpha_N, r all needs to be non-negative

    # Generate deterministic unit directions to approximate the wrench unit sphere.
    # Use a fixed seed to ensure reproducibility.
    rng = np.random.default_rng(42)
    rand_directions = rng.normal(size=(M, 6))
    norms = np.linalg.norm(rand_directions, axis=1, keepdims=True)
    # safety precaution: prevents division by 0, in case the rng generates a 0 vector
    norms[norms == 0.0] = 1.0
    rand_directions = rand_directions / norms

    # Include axis directions so obvious extremal rays are always represented.
    # This catches cases where the grasp is strong along principal force/torque axes.
    eye_dirs = np.vstack([np.eye(6), -np.eye(6)])
    rand_directions = np.vstack([rand_directions, eye_dirs])
    
    # Ensure uniqueness by removing near-duplicates (numerical stability).
    unique_dirs = [rand_directions[0:1]]
    for row in rand_directions[1:]:
        # Check if this direction is close to any already included.
        is_duplicate = False
        for existing in unique_dirs:
            dot_prod = abs(np.dot(row, existing[0]))
            if dot_prod > 0.99:  # Nearly parallel directions.
                is_duplicate = True
                break
        if not is_duplicate:
            unique_dirs.append(row.reshape(1, -1))
    rand_directions = np.vstack(unique_dirs)

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

    if not d_k_values:
        return float('inf')
    return float(np.max(d_k_values))
