import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from optimization.obstacles import CircularObstacle
from optimization.unicycle_planner import PlannerResult


def plot_trajectory(
    result: PlannerResult,
    obstacles: list[CircularObstacle],
    obstacle_buffer: float = 0.1,
) -> None:
    t_states = np.arange(len(result.x)) * result.dt
    t_controls = np.arange(len(result.v)) * result.dt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: XY path ---
    ax = axes[0]
    for obs in obstacles:
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius, color="gray", alpha=0.5))
        ax.add_patch(
            Circle(
                (obs.cx, obs.cy),
                obs.radius + obstacle_buffer,
                fill=False,
                linestyle="--",
                edgecolor="red",
            )
        )
    ax.plot(result.x, result.y, "b.-", linewidth=1.5, markersize=3, label="path")
    # Heading arrows (every few nodes)
    step = max(1, len(result.x) // 20)
    for i in range(0, len(result.x), step):
        ax.annotate(
            "",
            xy=(
                result.x[i] + 0.15 * np.cos(result.theta[i]),
                result.y[i] + 0.15 * np.sin(result.theta[i]),
            ),
            xytext=(result.x[i], result.y[i]),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
        )
    ax.plot(result.x[0], result.y[0], "go", markersize=10, label="start")
    ax.plot(result.x[-1], result.y[-1], "r*", markersize=14, label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("XY Path")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)

    # --- Panel 2: Controls ---
    ax = axes[1]
    ax.step(t_controls, result.v, where="post", label="v")
    ax.step(t_controls, result.omega, where="post", label="omega")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("control input")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True)

    # --- Panel 3: States ---
    ax = axes[2]
    ax.plot(t_states, result.x, label="x")
    ax.plot(t_states, result.y, label="y")
    ax.plot(t_states, result.theta, label="theta")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("state")
    ax.set_title("State Trajectories")
    ax.legend()
    ax.grid(True)

    status = "SOLVED" if result.success else "FAILED"
    fig.suptitle(f"Unicycle Trajectory ({status}, T={result.total_time:.3f}s)", fontsize=14)
    plt.tight_layout()
    plt.show()
