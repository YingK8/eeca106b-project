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
    ## TODO: for getting a viable trajectory on the Turtlebot you may need to edit these values or add entirely new parameters
    N: int = 300               # Increased so total time is 15.0 seconds!
    dt: float = 0.1
    v_min: float = 0.0         # Good, no reverse for tracking
    v_max: float = 0.2        # Physical limit of Turtlebot
    omega_min: float = -0.25    # Physical limit of Turtlebot
    omega_max: float = 0.25
    obstacle_buffer: float = 2
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.5]))
    # Add this line - DEFAULT P matrix for terminal cost, can be tuned separately if desired
    P: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 5.0]))

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
        xf = ca.DM(goal)
        Q = ca.DM(p.Q)
        R = ca.DM(p.R)
        P = ca.DM(p.P)  # Terminal cost weight (can be tuned separately if desired)

        # Scalar cost: sum over k of (x_k - xf)' Q (x_k - xf) + u_k' R u_k, plus terminal (x_N - xf)' P (x_N - xf)
        diff_x = X[:, :-1] - xf  # (3, N)
        state_cost = ca.sum1(ca.sum2(diff_x * (Q @ diff_x)))  # sum over k of (x_k - xf)' Q (x_k - xf)
        control_cost = ca.sum1(ca.sum2(U * (R @ U)))         # sum over k of u_k' R u_k
        term_diff = X[:, -1] - xf
        terminal_cost = (term_diff.T @ P) @ term_diff
        cost = state_cost + control_cost + terminal_cost

        opti.minimize(cost)

        ## TODO: Dynamics constraints — Euler integration (dt is fixed here)

        x_Np1 = X[0, :-1] + dt * U[0, :] * ca.cos(X[2, :-1])
        y_Np1 = X[1, :-1] + dt * U[0, :] * ca.sin(X[2, :-1])
        theta_Np1 = X[2, :-1] + dt * U[1, :]
        
        opti.subject_to(X[0, 1:] == x_Np1)
        opti.subject_to(X[1, 1:] == y_Np1)
        opti.subject_to(X[2, 1:] == theta_Np1)

        ## TODO: Boundary constraints — pin start and goal states
        xi = ca.DM(start)

        opti.subject_to(X[:, 0] == xi)
        opti.subject_to(X[:, -1] == xf)

        ## TODO: Control bounds — bound v and omega


        opti.subject_to(opti.bounded(p.v_min, U[0, :], p.v_max))
        opti.subject_to(opti.bounded(p.omega_min, U[1, :], p.omega_max))

        ## TODO: Obstacle avoidance — keep all nodes outside each obstacle

        for obs in obstacles:
            cx, cy = obs.cx, obs.cy
            r = obs.radius
            opti.subject_to((X[0,:] - cx) ** 2 + (X[1, :] - cy) ** 2 >= (r + p.obstacle_buffer) ** 2)

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
