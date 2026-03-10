from dataclasses import dataclass, field
import numpy as np
import casadi as ca

from optimization.obstacles import CircularObstacle

@dataclass
class PlannerParams:
    """Parameters for the minimum-time planner."""
    ## TODO: for getting a viable trajectory on the Turtlebot you may need to edit these values or add entirely new parameters
    N: int = 1000
    v_min: float = -5.0
    v_max: float = 5.0
    omega_min: float = -5.0
    omega_max: float = 5.0
    dt_min: float = 0.01
    dt_max: float = 1.0
    obstacle_buffer: float = 0.1


@dataclass
class TrackingParams:
    """Parameters for the quadratic-cost tracking planner."""
    ## TODO: for getting a viable trajectory on the Turtlebot you may need to edit these values or add entirely new parameters
    N: int = 300               # Increased so total time is 15.0 seconds!
    dt: float = 0.1
    v_min: float = -0.2        # Good, no reverse for tracking
    v_max: float = 0.2        # Physical limit of Turtlebot
    omega_min: float = -0.25    # Physical limit of Turtlebot
    omega_max: float = 0.25
    obstacle_buffer: float = 2
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.5]))
    # Add this line - DEFAULT P matrix for terminal cost, can be tuned separately if desired
    P: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 5.0]))
    # Force zero angular velocity for the first N steps to avoid an initial spin-in-place.
    num_no_turn_steps: int = 1
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
def generate_sine_guess(N, dt, amplitude=0.5, distance=2.0):
    """
    Generates a kinematic-feasible sine wave trajectory for an MPC warm start.
    N: Horizon length
    dt: Time step
    amplitude: How wide the curve is (adjust this to clear your obstacle!)
    distance: The 2-meter length of the maneuver
    """
    # 1. Generate x coordinates from 0 to distance
    x_guess = np.linspace(0, distance, N)
    
    # 2. Generate y coordinates using the sine function
    # At x=0, y=0. At x=2.0, y=sin(pi)=0.
    y_guess = amplitude * np.sin((np.pi / distance) * x_guess)
    
    # 3. Calculate heading (theta) based on the tangent of the curve
    # The derivative of the sine wave gives us the slope (dy/dx)
    dy_dx = amplitude * (np.pi / distance) * np.cos((np.pi / distance) * x_guess)
    theta_guess = np.arctan(dy_dx)
    
    # 4. Guess a constant forward velocity to finish in exactly N*dt seconds
    v_guess = np.full(N, distance / (N * dt))
    
    # 5. Calculate angular velocity (omega) as the rate of change of theta
    omega_guess = np.zeros(N)
    omega_guess[:-1] = np.diff(theta_guess) / dt
    omega_guess[-1] = omega_guess[-2] # copy last value to maintain array size
    
    return x_guess, y_guess, theta_guess, v_guess, omega_guess

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
        
        def generate_linear_guess(start, goal, N, dt):
            """
            Generates a straight-line initial guess from start to goal.
            States are size N+1, Controls are size N.
            """
            x0, y0, _ = start
            xf, yf, _ = goal
            
            # 1. Linear interpolation for X and Y (N+1 points)
            x_guess = np.linspace(x0, xf, N + 1)
            y_guess = np.linspace(y0, yf, N + 1)
            
            # 2. Constant heading pointing directly at the goal (N+1 points)
            heading = np.arctan2(yf - y0, xf - x0)
            theta_guess = np.full(N + 1, heading)
            
            # 3. Constant velocity to reach the goal exactly at the end of the horizon (N points)
            distance = np.hypot(xf - x0, yf - y0)
            v_guess = np.full(N, distance / (N * dt))
            
            # 4. Zero angular velocity for a straight line (N points)
            omega_guess = np.zeros(N)
            
            return x_guess, y_guess, theta_guess, v_guess, omega_guess

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
        # P = Q + A.T @ P @ A - A.T @ P @ B @ ca.inv(R + B.T @ P @ B) @ B.T @ P @ A  # LQR terminal cost
        #P = ca.DM(p.P)  # Terminal cost weight (can be tuned separately if desired)

        # cost = (X[:,:-1] - xf).T @ Q @ (X[:,:-1] - xf) + U.T @ R @ U + (X[:, -1] - xf).T @ P @ (X[:, -1] - xf) # Terminal cost with P matrix

        # cost = 0
        # for k in range(N):
        #     e = X[:, k] - xf
        #     u = U[:, k]
        #     # cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([u.T, R, u])
        #     cost += ca.mtimes(ca.mtimes(e.T, Q), e) + ca.mtimes(ca.mtimes(u.T, R), u)
        
        # eN = X[:, N] - xf
        # # cost += ca.mtimes([eN.T, P, eN])  # Terminal cost
        # cost += ca.mtimes(ca.mtimes(eN.T, P), eN)  # Terminal cost with P matrix
        # # import pdb; pdb.set_trace()
        diff_x = X[:, :-1] - xf  # (3, N)
        state_cost = ca.sum1(ca.sum2(diff_x * (Q @ diff_x)))  # sum over k of (x_k - xf)' Q (x_k - xf)
        control_cost = ca.sum1(ca.sum2(U * (R @ U)))         # sum over k of u_k' R u_k
        term_diff = X[:, -1] - xf
        terminal_cost = (term_diff.T @ P) @ term_diff
        cost = state_cost + control_cost + terminal_cost

        opti.minimize(cost)
        
        # opti.minimize(cost)

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

        ## No initial turn: force zero angular velocity for the first step(s)
        ## so the robot moves forward first instead of spinning in place.
        if p.num_no_turn_steps > 0:
            opti.subject_to(U[1, : p.num_no_turn_steps] == 0)

        for obs in obstacles:
            cx, cy = obs.cx, obs.cy
            r = obs.radius
            opti.subject_to((X[0,:] - cx) ** 2 + (X[1, :] - cy) ** 2 >= (r + p.obstacle_buffer) ** 2)

        # init guess
        x_g, y_g, th_g, v_g, w_g = generate_linear_guess(start, goal, N, dt)

        # Set the initial guess for the states
        opti.set_initial(X[0, :], x_g)
        opti.set_initial(X[1, :], y_g)
        opti.set_initial(X[2, :], th_g)

        # Set the initial guess for the controls
        opti.set_initial(U[0, :], v_g)
        opti.set_initial(U[1, :], w_g)
                                        
        

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 5000, "print_level": 5},
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
