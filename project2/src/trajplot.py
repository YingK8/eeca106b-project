import numpy as np


def trajplot(path, segment_time, coefficient):
    pathlength = path.shape[0]
    m = pathlength - 1
    time_vec = 0  # Keep track of segment_time over all segments
    trajectory = None

    ## Compute the trajectory in each segment.
    for i in range(m):
        ## Enter your code here
        start_time = time_vec
        time_vec = time_vec + segment_time[i]
        end_time = time_vec
        # check to see endtime is included
        t = np.linspace(start_time, end_time, num=11) # 11 points in each segment

        # Slicing the coefficient matrix 
        coeffs = coefficient[4*i : 4*(i+1), :] # shape (4,2)

        # stack coefficients with the time vector 
        T = np.array([np.ones_like(t), t, t**2, t**3]) # shape (4,11)

        position = T.T @ coeffs # shape (11, 4) @ (4,2) = (11,2)
        if trajectory is None:
            trajectory = position
        else:
            trajectory = np.vstack((trajectory, position)) # shape (11,2) + (N,2) = (N+11, 2)


    

    return trajectory
