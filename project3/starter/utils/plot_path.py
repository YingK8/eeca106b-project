import numpy as np
import matplotlib.pyplot as plt


def plot_path(map_data, path):
    """Plot the map with obstacles and the computed path.

    Parameters
    ----------
    map_data : dict
        Map data returned by load_map().
    path : np.ndarray, shape (M, 2) or empty
        Path coordinates. Each row is [x, y].
    """
    configuration = map_data['basicdata']
    resolution = map_data['resolution']
    num_rows = configuration.shape[0]

    fig, ax = plt.subplots()

    ## Plot boundary and obstacles
    for i in range(num_rows):
        x_down = configuration[i, 0]
        y_down = configuration[i, 1]
        x_up = configuration[i, 2]
        y_up = configuration[i, 3]

        X = [x_down, x_up, x_up, x_down]
        Y = [y_down, y_down, y_up, y_up]

        if i == 0:
            color = 'white'
            alpha = 0.1
        else:
            color = 'black'
            alpha = 0.5

        ax.fill(X, Y, color=color, alpha=alpha)

        ## Draw grid lines
        grid_x = np.arange(configuration[i, 0], configuration[i, 2] + resolution[0] / 2, resolution[0])
        grid_y = np.arange(configuration[i, 1], configuration[i, 3] + resolution[1] / 2, resolution[1])

        for gx in grid_x:
            ax.plot([gx, gx], [configuration[i, 1], configuration[i, 3]],
                    ':', color=[0.7, 0.7, 0.7], linewidth=0.5)
        for gy in grid_y:
            ax.plot([configuration[i, 0], configuration[i, 2]], [gy, gy],
                    ':', color=[0.7, 0.7, 0.7], linewidth=0.5)

    ## Plot the path if not empty
    if path is not None and len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], 'x-', color='red', linewidth=1.0)
        ax.plot(path[0, 0], path[0, 1], 'o', color='blue', markersize=8, linewidth=1.5)
        ax.plot(path[-1, 0], path[-1, 1], 'o', color='blue', markersize=8, linewidth=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.show()
