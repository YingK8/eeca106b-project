# Project 2: Trajectory Generation for a Point-Mass Robot

## Overview

In this project you will generate a smooth trajectory for a point-mass robot that passes through a series of waypoints. You are given a set of 2D waypoints and a total time budget. Your task is to fit cubic polynomials to each segment of the path so the robot smoothly travels through every waypoint.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

## File Structure

```
src/
├── paths/
│   ├── path1.txt          # 33 waypoints
│   └── path2.txt          # 32 waypoints
├── run_trajectory.py      # DO NOT MODIFY (below argparse)
├── trajectory_generator.py  # TODO
├── trajplot.py              # TODO
└── README.md
```

## What to Edit

You need to fill in code in **two files**: `trajectory_generator.py` and `trajplot.py`. Look for the `## Enter code here` comments.

### 1. `trajectory_generator.py`

This function takes a path (N x 2 array of waypoints) and a time tolerance (total time budget), and returns the polynomial coefficients and per-segment time allocations.

**First loop (line 33)** — Compute the Euclidean distance between consecutive waypoints:
- For each segment `i`, compute `distance[i] = ||path[i+1] - path[i]||`
- Hint: use `np.linalg.norm()`

**Second loop (line 38)** — For each segment `j`, you need to:
1. **Allocate time** proportionally: `segment_time[j] = distance[j] * time_tol / sum(distance)`
2. **Build a 4x4 constraint matrix** using the cubic polynomial basis evaluated at the segment's start time (`tinit`) and end time (`tend`):
   - Row 1: position at `tinit` → `[1, tinit, tinit^2, tinit^3]`
   - Row 2: velocity at `tinit` → `[0, 1, 2*tinit, 3*tinit^2]`
   - Row 3: position at `tend` → `[1, tend, tend^2, tend^3]`
   - Row 4: velocity at `tend` → `[0, 1, 2*tend, 3*tend^2]`
3. **Set the conditions** (right-hand side):
   - Initial position: `path[j]`
   - Initial velocity: `[0.3, 0.3]` (or `[0, 0]` for the first segment)
   - Final position: `path[j+1]`
   - Final velocity: `[0.3, 0.3]` (or `[0, 0]` for the last segment)
4. **Solve** for the 4x2 coefficient block: `coefficient = inv(constraints) @ conditions`
5. **Accumulate** `time_vec` for the next segment

Store each segment's 4x4 constraint block in `constraints[4*j:4*j+4, :]`, conditions in `conditions[4*j:4*j+4, :]`, and solved coefficients in `coefficient[4*j:4*j+4, :]`.

### 2. `trajplot.py`

This function evaluates the polynomial trajectory for plotting. For each segment `i`:

1. Compute `tinit` and `tend` from accumulated `time_vec` and `segment_time[i]`
2. Sample 11 evenly spaced points from `tinit` to `tend`
3. At each sample point `p`, evaluate the position using: `[1, p, p^2, p^3] @ coefficient[4*i:4*i+4, :]`
4. Append the segment's 11 points to `trajectory` using `np.vstack()`
5. Update `time_vec`

## Running

From the `src/` directory:

```bash
python run_trajectory.py --path paths/path1.txt --time 13
python run_trajectory.py --path paths/path2.txt --time 10
```

Run with `--help` to see all options:

```bash
python run_trajectory.py --help
```

## Expected Output

When complete, the plot should show a smooth curve passing through every waypoint. The robot starts and ends at rest (zero velocity) and maintains a cruising velocity of `[0.3, 0.3]` at intermediate waypoints.
