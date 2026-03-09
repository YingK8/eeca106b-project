#!/usr/bin/env python3
import argparse
import math

from optimization.obstacles import default_obstacle_scene, cory105_obstacle_scene
from optimization.unicycle_planner import (
    UnicyclePlanner, PlannerParams,
    UnicycleTrackingPlanner, TrackingParams,
)
from optimization.plot_trajectory import plot_trajectory
import numpy as np

def save_trajectory(result, filename):
    full_filename = filename + ".npz"
    np.savez(full_filename, x=result.x, y=result.y, theta=result.theta, v=result.v, omega=result.omega, total_time=result.total_time, dt=result.dt)
    print(f"Trajectory saved to {full_filename}")

def main():
    parser = argparse.ArgumentParser(description="Unicycle trajectory planner")
    parser.add_argument("--N", type=int, default=100, help="Number of discretization intervals")
    parser.add_argument(
        "--mode", choices=["min_time", "tracking"], default="min_time",
        help="Planner mode: 'min_time' (minimize T) or 'tracking' (quadratic LQR cost)",
    )
    parser.add_argument("--scene", type=str, default="default", help="Obstacle scene to use: 'default' or 'cory105'")
    parser.add_argument("--obstacle_buffer", type=float, default=0.1, help="Obstacle buffer")
    args = parser.parse_args()

    if args.scene == "default":
        start = (0.0, 0.0, 0.0)
        goal = (5.0, 5.0, math.pi / 2)
        obstacles = default_obstacle_scene()
    elif args.scene == "cory105":
        start = (0.0, 0.0, 0.0)
        goal = (2.5781, 0.0, 0.0)
        obstacles = cory105_obstacle_scene()

    if args.mode == "tracking":
        params = TrackingParams(N=args.N, obstacle_buffer=args.obstacle_buffer)
        planner = UnicycleTrackingPlanner(params)
        buf = params.obstacle_buffer
    else:
        params = PlannerParams(N=args.N, obstacle_buffer=args.obstacle_buffer)
        planner = UnicyclePlanner(params)
        buf = params.obstacle_buffer

    print(f"Solving with N={args.N}, mode={args.mode} ...")
    result = planner.solve(start, goal, obstacles)

    if result.success:
        print(f"Success! Total time T = {result.total_time:.4f} s, dt = {result.dt:.6f} s")
    else:
        print(f"Solver failed. Debug trajectory T = {result.total_time:.4f} s")

    plot_trajectory(result, obstacles, buf)

    save_trajectory(result, f"optimization_trajectory_{args.scene}_{args.mode}")

if __name__ == "__main__":
    main()
