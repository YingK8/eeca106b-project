#!/usr/bin/env python3
import argparse
import math

from optimization.obstacles import default_obstacle_scene
from optimization.unicycle_planner import (
    UnicyclePlanner, PlannerParams,
    UnicycleTrackingPlanner, TrackingParams,
)
from optimization.plot_trajectory import plot_trajectory


def main():
    parser = argparse.ArgumentParser(description="Unicycle trajectory planner")
    parser.add_argument("--N", type=int, default=100, help="Number of discretization intervals")
    parser.add_argument(
        "--mode", choices=["min_time", "tracking"], default="min_time",
        help="Planner mode: 'min_time' (minimize T) or 'tracking' (quadratic LQR cost)",
    )
    args = parser.parse_args()

    start = (0.0, 0.0, 0.0)
    goal = (5.0, 5.0, math.pi / 2)
    obstacles = default_obstacle_scene()

    if args.mode == "tracking":
        params = TrackingParams(N=args.N)
        planner = UnicycleTrackingPlanner(params)
        buf = params.obstacle_buffer
    else:
        params = PlannerParams(N=args.N)
        planner = UnicyclePlanner(params)
        buf = params.obstacle_buffer

    print(f"Solving with N={args.N}, mode={args.mode} ...")
    result = planner.solve(start, goal, obstacles)

    if result.success:
        print(f"Success! Total time T = {result.total_time:.4f} s, dt = {result.dt:.6f} s")
    else:
        print(f"Solver failed. Debug trajectory T = {result.total_time:.4f} s")

    plot_trajectory(result, obstacles, buf)


if __name__ == "__main__":
    main()
