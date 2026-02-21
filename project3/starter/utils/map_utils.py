import numpy as np


def worldtocell(leftbound, resolution, point):
    """Convert world coordinates to grid cell indices (0-based).

    Parameters
    ----------
    leftbound : array-like, shape (2,)
        Lower-left corner of the map boundary [x_min, y_min].
    resolution : array-like, shape (2,)
        Grid resolution [xy_res, xy_res].
    point : array-like, shape (2,)
        World coordinates [x, y].

    Returns
    -------
    np.ndarray, shape (2,)
        Cell indices [nx, ny] (0-based).
    """
    return np.floor((np.array(point) - np.array(leftbound)) / np.array(resolution)).astype(int)


def celltoworld(leftbound, resolution, point):
    """Convert grid cell indices (0-based) to world coordinates (cell center).

    Parameters
    ----------
    leftbound : array-like, shape (2,)
        Lower-left corner of the map boundary [x_min, y_min].
    resolution : array-like, shape (2,)
        Grid resolution [xy_res, xy_res].
    point : array-like, shape (2,)
        Cell indices [nx, ny] (0-based).

    Returns
    -------
    np.ndarray, shape (2,)
        World coordinates [x, y] at the center of the cell.
    """
    leftbound = np.array(leftbound)
    resolution = np.array(resolution)
    point = np.array(point)
    xy_res = resolution[0]
    nx = point[0]
    ny = point[1]
    return np.array([
        xy_res / 2 + nx * xy_res + leftbound[0],
        xy_res / 2 + ny * xy_res + leftbound[1],
    ])


def celltonumber(segment, point):
    """Convert 2D cell indices (0-based) to a 1D node number (0-based).

    Parameters
    ----------
    segment : array-like, shape (2,)
        Grid dimensions [mx, my].
    point : array-like, shape (2,)
        Cell indices [nx, ny] (0-based).

    Returns
    -------
    int
        Node number (0-based).
    """
    mx = int(segment[0])
    return int(point[0]) + int(point[1]) * mx


def numbertocell(segment, number):
    """Convert a 1D node number (0-based) to 2D cell indices (0-based).

    Parameters
    ----------
    segment : array-like, shape (2,)
        Grid dimensions [mx, my].
    number : int
        Node number (0-based).

    Returns
    -------
    np.ndarray, shape (2,)
        Cell indices [nx, ny] (0-based).
    """
    mx = int(segment[0])
    ny = number // mx
    nx = number % mx
    return np.array([nx, ny])
