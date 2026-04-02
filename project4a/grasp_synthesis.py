import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import mujoco as mj
import grasp_synthesis
import types
from utils import *

# [MT]
import scipy as scp
import casadi as ca

# debug flag
DEBUG_FLAG  = False
# optimizer level
OPT_LEVEL   = 0



def synthesize_grasp(env: grasp_synthesis.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=1000, 
                        #  max_iters=100, 
                         lr=0.1):
    # Optimize hand joints for force closure grasp
    q_h = q_h_init.copy()
    for it in range(max_iters):
        in_contact = False  # Check if enough contacts
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
        
        # Compute objective and gradient
        fval = joint_space_objective(env, q_h, fingertip_names, in_contact)
        grad = numeric_gradient(joint_space_objective, q_h, env, fingertip_names, in_contact)

        # Gradient descent step
        q_h_new = q_h.copy() - lr*grad
        
        # Clip to valid joint limits
        q_h_new = clip_to_valid_state_slice(env.physics, q_h_new, env.q_h_slice)

        # Check improvement
        fval_new = joint_space_objective(env, q_h_new, fingertip_names, in_contact)

        # Accept if improved
        if fval_new < fval:
            q_h = q_h_new
            improvement = fval - fval_new 
            print(f"Iter {it}, objective={fval_new:.7f}, improvement={improvement:.7f}")
            if improvement < 1e-6:
                break
        else:
            lr *= 0.5  # Reduce learning rate if no improvement
            print(f"Iter {it}, no improvement, reduce lr to {lr}")
            if lr < 1e-6:
                break
    return q_h

def joint_space_objective(env: grasp_synthesis.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10, 
                          friction_coeff=0.5, 
                        #   num_friction_cone_approx=4,
                          num_friction_cone_approx=8,  # [MT]
                        #   eps=0.000001,
                          eps=0.00001,
                          ):
    # Objective: penalize distance from surface and lack of force closure
    env.set_configuration(q_h)
    finger_positions = env.get_body_positions(fingertip_names)

    # Penalize distance from object surface
    surface_penalty = 0.0
    for p in finger_positions:
        d = env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)
        surface_penalty += d*d
    if not in_contact or env.physics.data.ptr.contact.frame.shape[0] < 4:
        return beta * surface_penalty  # Return penalty if not in contact
    else:
        # Build friction cones

        # # to obtain the id correspondence of contact frame and ids
        # geom_id_pairs_ = env.physics.data.ptr.contact.geom
        # indices_ = [
        #     i_
        #     for i_, pair_ in enumerate(geom_id_pairs_)
        #     if "table/table_geom" not in
        #         [mj.mj_id2name(env.physics.model.ptr, mj.mjtObj.mjOBJ_GEOM, gid) for gid in pair_]
        #     and (
        #         mj.mj_id2name(env.physics.model.ptr, mj.mjtObj.mjOBJ_GEOM, pair_[0]) == env.object_name
        #         or
        #         mj.mj_id2name(env.physics.model.ptr, mj.mjtObj.mjOBJ_GEOM, pair_[1]) == env.object_name
        #         )
        #     ]

        contact_frames, contact_positions = env.get_contact_normals_and_positions(env.physics.data.ptr.contact)
        

        # Build directions for each contact
        directions_list = []
        for i in range(len(contact_frames)):
            contact_frame = contact_frames[i]
            directions_i = build_friction_cone(contact_frame, friction_coeff, num_friction_cone_approx)
            directions_list.append(directions_i)

        # Build grasp matrix
        G = build_grasp_matrix(contact_positions, directions_list, origin=env.sphere_center)

        # Optimize Q+ and Q- distances
        Q_plus_dist = optimize_necessary_condition(G, env)
        if Q_plus_dist > eps:
            score_fc = Q_plus_dist
        else:
            Q_minus_dist = optimize_sufficient_condition(G)
            score_fc = Q_minus_dist
        return score_fc + beta * surface_penalty


def build_friction_cone(normal: np.array, mu=0.5, num_approx=4) -> list:
    # Build friction cone vectors for a contact
    
    # define the angles for boundary vector
    theta = 2.0 * np.pi / num_approx
    z_axis = normal[0:3]
    y_axis = normal[6:9]
    x_axis = normal[3:6]
    
    boundary_vecs = []
    for i_ in range(num_approx):
        theta_i = i_ * theta
        vec = mu * (np.cos(theta_i) * x_axis + np.sin(theta_i) * y_axis) + z_axis
        vec /= np.linalg.norm(vec)
        boundary_vecs.append(vec)
    return boundary_vecs


def build_grasp_matrix(positions: np.array, friction_cones: list, origin=np.zeros(3)):
    # Build grasp matrix from contact positions and friction cones
    #YOUR CODE HERE
    ### G^j_i = [         vec^j_i       ]
    ###         [ (pos-ori)_i x vec^j_i ]

    G_mat = np.empty((6,0))
    n_contacts = len(positions)
    for i_ in range(n_contacts):
        pos_vec = positions[i_,:]
        for j_ in range(len(friction_cones[i_])):
            dir_vec = friction_cones[i_][j_]
            G_ji = np.hstack((dir_vec, np.cross((pos_vec - origin), dir_vec)))
            G_mat = np.hstack((G_mat, G_ji.reshape(6,1)))
    return G_mat


def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv):
    # Solve Q+ optimization (L2 distance)
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
        {"max_iter": 3000, "print_level": OPT_LEVEL, "sb": "yes"}
    )

    # solve the optimization problem
    sol = opti.solve()
    
    # obtain the optimization output
    alpha_vec = np.array(sol.value(X[:])).flatten().reshape(N_,1)

    return np.sqrt( (alpha_vec.T @ G.T) @ (G @ alpha_vec) ).squeeze()
    # return ( (alpha_vec.T @ G.T) @ (G @ alpha_vec)).squeeze()



def optimize_sufficient_condition(G: np.array, M=20):
    # Solve Q- optimization (max ball radius)
    # extract dimensions
    D_, N_ = G.shape        # dimension of workspace and number of friction cone boundary vectors
    q_M_arr = sampleRandomPointOnHypersphere(dim=D_, num_pts=M).T  # (D_, M)

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
        {"max_iter": 3000, "print_level": OPT_LEVEL}
    )

    # optimization problem may not be always well constructed, using try-except
    try:
        sol = opti.solve()
        return -float(sol.value(r)) # notice the negative sign
    except Exception as err:
        if DEBUG_FLAG:
            print(err)
    
    return 0.0  # return 0.0 radius if no valid solution
