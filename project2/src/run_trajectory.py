import argparse
import numpy as np
import matplotlib.pyplot as plt
from trajectory_generator import trajectory_generator
from trajplot import trajplot

parser = argparse.ArgumentParser(description='Trajectory Generation for a Point-Mass Robot')
parser.add_argument('--path', type=str, default='paths/path1.txt',
                    help='Path to the waypoints file (default: paths/path1.txt)')
parser.add_argument('--time', type=float, default=13.0,
                    help='Total trajectory time budget in seconds (default: 13.0)')
args = parser.parse_args()

## Plan path
print('Trajectory Generation ...')

## Load the given path from txt
path = np.loadtxt(args.path, delimiter='\t')

## DO NOT MODIFY BELOW THIS LINE
## Run trajectory
coefficient, segment_time = trajectory_generator(path, args.time)

## Plot trajectory
trajectory = trajplot(path, segment_time, coefficient)

plt.plot(path[:, 0], path[:, 1], linestyle='none', marker='x', label='Waypoints')
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory Generation')
plt.legend()
plt.show()
