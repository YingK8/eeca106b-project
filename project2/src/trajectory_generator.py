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
    constraints = np.zeros((4 * m, 4))
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
        pass

    ## Compute the coefficient matrix
    for j in range(m):
        ## Enter code here
        pass

    return coefficient, segment_time
