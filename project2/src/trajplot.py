import numpy as np


def trajplot(path, segment_time, coefficient):
    pathlength = path.shape[0]
    m = pathlength - 1
    time_vec = 0  # Keep track of segment_time over all segments
    trajectory = np.empty((0, 2))

    ## Compute the trajectory in each segment.
    for i in range(m):
        ## Enter your code here
        pass

    return trajectory
