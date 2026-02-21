from dataclasses import dataclass, field
import numpy as np
import casadi as ca

from optimization.obstacles import CircularObstacle

@dataclass
class PlannerParams:
    """Parameters for the minimum-time planner."""
    N: int = 100
    v_min: float = -0.1
    v_max: float = 1.0
    omega_min: float = -2.0
    omega_max: float = 2.0
    dt_min: float = 0.01
    dt_max: float = 1.0
    obstacle_buffer: float = 0.1


@dataclass
class TrackingParams:
    """Parameters for the quadratic-cost tracking planner."""
    N: int = 100
    dt: float = 0.1
    v_min: float = -0.1
    v_max: float = 1.0
    omega_min: float = -2.0
    omega_max: float = 2.0
    obstacle_buffer: float = 0.1
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.5]))


@dataclass
class PlannerResult:
    success: bool
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    omega: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: float = 0.0
    total_time: float = 0.0
    solver_stats: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Option 1: Minimum-time planner
# ──────────────────────────────────────────────────────────────────────────────

class UnicyclePlanner:
    """Minimum-time trajectory planner for a unicycle robot.

    Decision variables:
        X : (3, N+1) — state trajectory [x; y; theta] at each node
        U : (2, N)   — control inputs [v; omega] at each interval
        T : scalar   — total trajectory time
        dt = T / N   — derived per-step timestep
    """

    def __init__(self, params: PlannerParams | None = None):
        self.params = params or PlannerParams()

    def solve(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: list[CircularObstacle] | None = None,
    ) -> PlannerResult:
        p = self.params
        N = p.N
        obstacles = obstacles or []

        opti = ca.Opti()

        ## Decision variables
        X = opti.variable(3, N + 1)  # [x; y; theta] at each node
        U = opti.variable(2, N)      # [v; omega] at each interval
        T = opti.variable()           # total trajectory time
        dt = T / N                    # derived timestep

        ## TODO: Objective — minimize total trajectory time

        ## TODO: Dynamics constraints — Euler integration of unicycle model

        ## TODO: Boundary constraints — pin start and goal states

        ## TODO: Control bounds — bound v and omega

        ## TODO: Time bounds — bound total time T (Force to be positive)

        ## TODO: Obstacle avoidance — keep all nodes outside each obstacle

        ## TODO: Initial guess for T

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 3000, "print_level": 5},
        )

        try:
            sol = opti.solve()
            T_sol = float(sol.value(T))
            return PlannerResult(
                success=True,
                x=np.array(sol.value(X[0, :])).flatten(),
                y=np.array(sol.value(X[1, :])).flatten(),
                theta=np.array(sol.value(X[2, :])).flatten(),
                v=np.array(sol.value(U[0, :])).flatten(),
                omega=np.array(sol.value(U[1, :])).flatten(),
                dt=T_sol / N,
                total_time=T_sol,
                solver_stats=sol.stats(),
            )
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            debug = opti.debug
            T_dbg = float(debug.value(T))
            return PlannerResult(
                success=False,
                x=np.array(debug.value(X[0, :])).flatten(),
                y=np.array(debug.value(X[1, :])).flatten(),
                theta=np.array(debug.value(X[2, :])).flatten(),
                v=np.array(debug.value(U[0, :])).flatten(),
                omega=np.array(debug.value(U[1, :])).flatten(),
                dt=T_dbg / N,
                total_time=T_dbg,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Option 2: Quadratic tracking-cost planner
# ──────────────────────────────────────────────────────────────────────────────

class UnicycleTrackingPlanner:
    """Fixed-horizon quadratic tracking cost planner.

    Decision variables:
        X : (3, N+1) — state trajectory [x; y; theta] at each node
        U : (2, N)   — control inputs [v; omega] at each interval

    """

    def __init__(self, params: TrackingParams | None = None):
        self.params = params or TrackingParams()

    def solve(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: list[CircularObstacle] | None = None,
    ) -> PlannerResult:
        p = self.params
        N = p.N
        dt = p.dt
        obstacles = obstacles or []

        opti = ca.Opti()

        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)

        ## TODO: Objective — quadratic tracking cost with terminal penalty

        ## TODO: Dynamics constraints — Euler integration (dt is fixed here)

        ## TODO: Boundary constraints — pin start and goal states

        ## TODO: Control bounds — bound v and omega

        ## TODO: Obstacle avoidance — keep all nodes outside each obstacle

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 3000, "print_level": 5},
        )

        total_time = N * dt
        try:
            sol = opti.solve()
            return PlannerResult(
                success=True,
                x=np.array(sol.value(X[0, :])).flatten(),
                y=np.array(sol.value(X[1, :])).flatten(),
                theta=np.array(sol.value(X[2, :])).flatten(),
                v=np.array(sol.value(U[0, :])).flatten(),
                omega=np.array(sol.value(U[1, :])).flatten(),
                dt=dt,
                total_time=total_time,
                solver_stats=sol.stats(),
            )
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            debug = opti.debug
            return PlannerResult(
                success=False,
                x=np.array(debug.value(X[0, :])).flatten(),
                y=np.array(debug.value(X[1, :])).flatten(),
                theta=np.array(debug.value(X[2, :])).flatten(),
                v=np.array(debug.value(U[0, :])).flatten(),
                omega=np.array(debug.value(U[1, :])).flatten(),
                dt=dt,
                total_time=total_time,
            )
