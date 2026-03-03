import heapq

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

    pq  = []  # Priority queue for Dijkstra's algorithm
    dist = {node: float('inf') for node in nodenumber}  # Distance from start to each node
    prev = [-1] * num_nodes  # Previous node in optimal path initialization

    # Cell indice of the start position in the grid / cell frame
    start_cell = worldtocell(leftbound, xy_res, start)
    goal_cell = worldtocell(leftbound, xy_res, goal)

    start_node = celltonumber(segment ,start_cell)
    goal_node = celltonumber(segment, goal_cell)

    dist[start_node] = 0
    pq.append((0, start_node))  # (distance, node)
    final_cost = 0
    completed = False
    while pq:
        current_dist, current_node = heapq.heappop(pq)  # Get node with smallest distance
        if current_node == goal_node:
            final_cost = current_dist
            completed = True
            break

        cell = numbertocell([mx, my], current_node)
        neighbors = [
            (cell[0] + dx, cell[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        valid_neighbors = []
        for neighbor in neighbors:
            if 0 <= neighbor[0] < mx and 0 <= neighbor[1] < my:
                neighbor_node = celltonumber(segment, neighbor)
                if blockflag[neighbor_node] == 0:  # Check if the neighbor is not blocked
                    valid_neighbors.append(neighbor_node)
        
        for neighbor_node in valid_neighbors:
            alt_dist = dist[current_node] + xy_res # Adjacent cost to the neighbor
            if alt_dist < dist[neighbor_node]:
                dist[neighbor_node] = alt_dist
                prev[neighbor_node] = current_node
                heapq.heappush(pq, (alt_dist, neighbor_node))
            

        # for node in nodenumber:
        #     if node == celltonumber(start_cell, mx):
        #         dist[node] = 0
        #         pq.append((0, node))  # (distance, node)

    path = []
    
    if not completed:
        return np.array(path)  # Return empty path if no path is found
    current_node = goal_node
    while current_node != -1:
        cell = numbertocell([mx, my], current_node)
        world_pos = celltoworld(leftbound, resolution, cell)
        path.append(world_pos)
        current_node = prev[current_node]
    
    path.reverse()  # Reverse the path to get it from start to goal
    return np.array(path)





    # return final_cost

    # path = np.array([]).reshape(0, 2)
    # return path
