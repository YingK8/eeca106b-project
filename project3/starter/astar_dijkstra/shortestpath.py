import numpy as np
from utils.map_utils import worldtocell, celltoworld, celltonumber, numbertocell


def shortestpath(map_data, start, goal):
    """Find the shortest path from start to goal using Dijkstra's algorithm.

    Parameters
    ----------
    map_data : dict
        Map data returned by load_map().
    start : np.ndarray, shape (2,)
        Start position in world coordinates [x, y].
    goal : np.ndarray, shape (2,)
        Goal position in world coordinates [x, y].

    Returns
    -------
    np.ndarray, shape (M, 2)
        Path from start to goal. Each row is [x, y]. Returns empty array if
        no path is found.
    """
    ## DO NOT MODIFY
    nodenumber = map_data['nodenumber']
    leftbound = map_data['boundary'][:2]
    blockflag = map_data['blockflag']
    resolution = map_data['resolution']
    xy_res = resolution[0]
    segment = map_data['segment']
    mx, my = int(segment[0]), int(segment[1])
    num_nodes = len(nodenumber)

    ## Enter code here

    path = np.array([]).reshape(0, 2)
    return path
