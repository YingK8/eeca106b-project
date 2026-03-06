import numpy as np


def trajectory_generator(path, time_tol):
    """Turn a given path into a trajectory.

    NOTE: The input to the function is the given path which is a matrix of
    size (N x 2), where N is the total no. of points.
    The function outputs the coefficient matrix and time segments.
    """
    pathlength = path.shape[0]
    m = pathlength - 1

    # The matrix that describes all the constraints of the system is named as
    # constraints in this file
    constraints = np.zeros((4 * m, 4*m))
    conditions = np.zeros((4 * m, 2))
    coefficient = np.zeros((4 * m, 2))
    segment_time = np.zeros(m)

    # sample constraints form
    # x    = a0 + a1*t + a2*t^2 + a3*t^3     ; position
    # x'   = 0  + a1   + 2*a2*t + 3*a3*t^2   ; velocity
    # x''  = 0  + 0    + 2*a2   + 6*a3*t     ; acceleration

    # sample conditions form
    # x    = a0 + a1*t + a2*t^2 + a3*t^3                           ; position
    # x'   = 0  + a1   + 2*a2*t + 3*a3*t^2                         ; velocity
    # constraint conditions = constraints * coefficient matrix
    # constraints and constraints condition for the start position

    ## Compute the distance between segments
    for i in range(m):
        ## Enter code here
        d_i = np.linalg.norm(path[i+1] - path[i])
        segment_time[i] = d_i # *temporarily* store the distances in the segment_time

    total_segment_lengths = np.sum(segment_time)
    segment_time = segment_time * time_tol / total_segment_lengths # finishes computing the segment_times

    ## Compute the coefficient matrix
    T = 0
    for j in range(m):
        ## Enter code here
        t_i = T
        T = T + segment_time[j]
        t_f = T

        # calculating the constraints matrix
        M = np.matrix(
            [
                [1, t_i, t_i**2, t_i**3],       # inital pos
                [0, 1, 2 * t_i, 3 * (t_i**2)],  # inital vel
                [1, t_f, t_f**2, t_f**3],       # final pos
                [0, 1, 2 * t_f, 3 * (t_f**2)],  # final vel
            ]
        )
        constraints[4*j : 4*(j+1), :] = M

        # calculating the conditions matrix (path is length m+1)
        x_i, y_i = path[j]
        x_f, y_f = path[j+1]
        C = np.matrix(
            [
                [x_i, y_i],     # inital pos
                [0.3, 0.3],     # inital vel
                [x_f, y_f],     # final pos
                [0.3, 0.3],     # final vel
            ]
        )

        Coeff = np.linalg.solve(M, C) 


        conditions[4*j : 4*(j+1), :] = C
    
    if len(conditions) < 2:
        # Errored response
        print("Failed because there is not enough points that are registered. The conditions matrix is less than 2. ")
        return np.zeros((4*m)), segment_time
    conditions[1] = np.array([0, 0]) # set the initial velocity to zero
    conditions[-2] = np.array([0, 0]) # set the final velocity
    
    # solve for the coefficient matrix
    coefficient = np.linalg.solve(constraints, conditions)  

    return coefficient, segment_time
