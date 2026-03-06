import argparse
import numpy as np
from utils.load_map import load_map
from shortestpath import shortestpath
from utils.plot_path import plot_path
from astar import astar

def main():
    parser = argparse.ArgumentParser(description='Path Planning with Dijkstra/A*')
    parser.add_argument('--map', type=str, default='utils/maps/map1.txt',
                        help='Path to the map file (default: utils/maps/map1.txt)')
    parser.add_argument('--res', type=float, default=0.1,
                        help='Grid resolution (default: 0.1)')
    parser.add_argument('--margin', type=float, default=0.25,
                        help='Obstacle inflation margin (default: 0.25)')
    parser.add_argument('--start', type=float, nargs=2, default=[2.0, 4.5],
                        help='Start position x y (default: 2.0 4.5)')
    parser.add_argument('--goal', type=float, nargs=2, default=[19.0, 3.0],
                        help='Goal position x y (default: 19.0 3.0)')
    parser.add_argument('--algo', type=str,
                        choices=['dijkstra', 'astar'],
                        help='Algorithm to use (default: dijkstra)')
    args = parser.parse_args()

    ## DO NOT MODIFY BELOW THIS LINE
    print('Planning ...')

    map_data = load_map(args.map, args.res, args.margin)
    start = np.array(args.start)
    goal = np.array(args.goal)

    if args.algo == 'dijkstra':
        path = shortestpath(map_data, start, goal)
    elif args.algo == 'astar':
        path = astar(map_data, start, goal)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    plot_path(map_data, path)

if __name__ == "__main__":
    main()
