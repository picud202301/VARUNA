# ===============================================================================
# SolverZermeloPSO — Particle Swarm Optimization + Analytic Guidance
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.5
# Description:
#   Heuristic solver for Zermelo's navigation problem that combines:
#     - Analytic guidance (optional single intermediate waypoint)
#     - Particle Swarm Optimization (global search over angular-rate profile)
#     - Local refinement with SLSQP (SciPy)
#
#   Updates in this version:
#     - SafePolynomialMutation implemented
#     - Monkey patch applied directly to PolynomialMutation._do
#     - RuntimeWarnings from pymoo eliminated
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import time
import traceback
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
from numba import njit
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from scipy.optimize import minimize as scipy_minimize
from problems.zermelo.solvers.SolverZermeloAnalytic import SolverZermeloAnalytic
import pymoo.operators.mutation.pm as pm


# -------------------------------------------------------------------------------
# Termination: time limit or no improvement for N iterations
# -------------------------------------------------------------------------------
class TimeOrNoImprovement(Termination):
    """
    Termination criterion that stops when a wall-clock time limit is reached or when
    there is no improvement in the best objective for a fixed number of iterations.

    Parameters
    ----------
    max_time : float | None, optional
        Maximum allowed time in seconds. If None, time limit is disabled.
    n_last : int, optional
        Number of last iterations to check for improvement.
    tol : float, optional
        Minimum required improvement to reset the no-improvement counter.
    logger : Callable[..., None] | None, optional
        Logger function for termination messages.

    Attributes
    ----------
    start_time : float | None
        Start time (perf_counter) when the optimization begins.
    no_improve_counter : int
        Counter for consecutive iterations without sufficient improvement.
    best : float | None
        Best (lowest) objective value observed so far.
    """

    def __init__(self, max_time: Optional[float] = None, n_last: int = 50, tol: float = 1e-6,
                 logger: Optional[Callable[..., None]] = None) -> None:
        try:
            super().__init__()
            self.max_time = max_time
            self.n_last = n_last
            self.tol = tol
            self.start_time: Optional[float] = None
            self.no_improve_counter: int = 0
            self.best: Optional[float] = None
            self.logger = logger or (lambda *a, **k: None)
        except Exception as e:
            print(f"[ERROR] TimeOrNoImprovement.__init__ failed: {e}")
            raise

    def _update(self, algorithm) -> bool:
        """
        Internal pymoo hook to decide termination.

        Parameters
        ----------
        algorithm : Any
            The current algorithm instance from pymoo.

        Returns
        -------
        bool
            True to terminate, False to continue.

        Raises
        ------
        Exception
            Propagates any error during state inspection.
        """
        try:
            if self.start_time is None:
                self.start_time = time.perf_counter()

            if self.max_time is not None and (time.perf_counter() - self.start_time) >= float(self.max_time):
                self.logger("[TERM] Time limit reached.")
                return True

            opt = getattr(algorithm, "opt", None)
            if opt is None or len(opt) == 0:
                return False

            current_best = float(np.atleast_1d(opt.get("F")).ravel()[0])
            if self.best is None or (self.best - current_best) > self.tol:
                self.best = current_best
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1
                if self.no_improve_counter >= self.n_last:
                    self.logger(f"[TERM] No improvement for {self.n_last} iterations.")
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] TimeOrNoImprovement._update failed: {e}")
            raise


# -------------------------------------------------------------------------------
# JIT helper: expand control points to full horizon by linear interpolation
# -------------------------------------------------------------------------------
@njit(cache=True)
def interpolateControls(u_values: np.ndarray, max_steps: int) -> np.ndarray:
    """
    Expand a vector of control points into a full-length control sequence via linear interpolation.

    Notes
    -----
    This function is JIT-compiled with Numba for speed. Do not introduce Python
    exception handling here to preserve nopython performance.

    Parameters
    ----------
    u_values : np.ndarray
        Control points (angular rates).
    max_steps : int
        Desired horizon length.

    Returns
    -------
    np.ndarray
        Interpolated control sequence of length `max_steps`.
    """
    n_control_points = u_values.size
    if max_steps <= 1 or n_control_points == 1:
        return np.full(max_steps, u_values[0] if n_control_points > 0 else 0.0)
    positions = np.linspace(0.0, n_control_points - 1.0, max_steps)
    indices = np.floor(positions).astype(np.int64)
    next_indices = np.minimum(indices + 1, n_control_points - 1)
    t = positions - indices
    return (1.0 - t) * u_values[indices] + t * u_values[next_indices]


# -------------------------------------------------------------------------------
# Safe mutation patch (directly override PolynomialMutation._do)
# -------------------------------------------------------------------------------
def safeDo(self, problem, X, **kwargs):
    """
    Robust replacement for pymoo's PolynomialMutation._do that guards against
    NaNs/Infs and enforces bounds elementwise.

    Parameters
    ----------
    self : pm.PolynomialMutation
        Mutation operator instance.
    problem : pymoo.core.problem.Problem
        Pymoo problem providing bounds via problem.bounds().
    X : np.ndarray
        Population array (n_individuals, n_vars).

    Returns
    -------
    np.ndarray
        Mutated population array.
    """
    try:
        X = X.astype(float)
        xl, xu = problem.bounds()
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)

        # Ensure prob and eta are numeric scalars
        prob = float(self.prob.get()) if hasattr(self.prob, "get") else float(self.prob or 1.0)
        eta = float(self.eta.get()) if hasattr(self.eta, "get") else float(self.eta or 20)

        is_mut = np.random.random(X.shape) < prob
        rand = np.random.random(X.shape)

        delta1 = np.clip((X - xl) / (xu - xl), 0.0, 1.0)
        delta2 = np.clip((xu - X) / (xu - xl), 0.0, 1.0)
        mut_pow = 1.0 / (eta + 1.0)

        deltaq = np.where(
            delta1 < delta2,
            np.power(2.0 * rand + (1.0 - 2.0 * rand) * np.power(1.0 - delta1, eta + 1.0), mut_pow) - 1.0,
            1.0 - np.power(
                2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(1.0 - delta2, eta + 1.0),
                mut_pow,
            ),
        )

        _X = X + deltaq * (xu - xl)
        _X = np.clip(_X, xl, xu)
        _X = np.nan_to_num(_X, nan=0.0, posinf=xu, neginf=xl)

        X[is_mut] = _X[is_mut]
        return X
    except Exception as e:
        print(f"[ERROR] safeDo failed: {e}")
        raise


pm.PolynomialMutation._do = safeDo
# print(">>> Patched PolynomialMutation._do with safeDo <<<")


# -------------------------------------------------------------------------------
# Solver
# -------------------------------------------------------------------------------
class SolverZermeloPSO(SolverZermeloAnalytic):
    """
    PSO-based heuristic solver with optional analytic guidance and local SLSQP refinement.

    The solver searches over piecewise-linear angular-rate profiles (expanded to the full
    horizon) while simulating vessel dynamics, optionally seeding from analytic segments.

    Attributes
    ----------
    name : str
        Human-readable solver name ("PSO").
    id : int
        Solver identification code.
    controls_history : list[tuple[float, float]] | None
        Expanded control sequence [(v, r), ...] used in the final simulation.
    n_control_points : int
        Number of control points composing the angular-rate profile.
    n_particles : int
        PSO population size.
    print_enabled : bool
        Enable log printing to stdout.
    max_steps : int | None
        Maximum number of integration steps (horizon).
    time_step : float | None
        Integration step size.
    ship_velocity : float | None
        Vessel speed in still water (from scenario).
    goal : np.ndarray | None
        Goal coordinates.
    start : np.ndarray | None
        Start coordinates.
    max_scenario_size : float | None
        Max dimension of the scenario, used to adapt PSO hyperparameters.
    """

    class _ZermeloProblem(Problem):
        """
        Pymoo single-objective problem wrapper for PSO evaluation.

        Notes
        -----
        The decision variables are angular-rate control points bounded by
        [r_min + eps, r_max - eps]. Objective is the solver-level fitness.
        """

        def __init__(self, solver: "SolverZermeloPSO") -> None:
            try:
                epsilon = 1e-8
                xl = np.full(solver.n_control_points, solver.scenario.getRMin() + epsilon)
                xu = np.full(solver.n_control_points, solver.scenario.getRMax() - epsilon)
                super().__init__(n_var=solver.n_control_points, n_obj=1, n_constr=0, xl=xl, xu=xu)
                self.solver = solver
                self.id = 2  # Solver identification
            except Exception as e:
                print(f"[ERROR] SolverZermeloPSO._ZermeloProblem.__init__ failed: {e}")
                raise

        def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
            """
            Vectorized evaluation of the population.

            Parameters
            ----------
            X : np.ndarray
                Population array (n_individuals, n_vars).
            out : dict
                Output dict where 'F' will be stored as (n_individuals, 1).
            """
            try:
                out["F"] = np.array([self.solver.fitness(x) for x in X]).reshape(-1, 1)
            except Exception as e:
                print(f"[ERROR] SolverZermeloPSO._ZermeloProblem._evaluate failed: {e}")
                raise

    def __init__(self, scenario: object, rng: np.random.Generator, parameters: dict[str, Any]) -> None:
        """
        Initialize the PSO-based solver with optional analytic guidance.

        Parameters
        ----------
        scenario : object
            Scenario instance exposing geometry, currents, and goal.
        rng : np.random.Generator
            Random number generator for reproducibility.
        parameters : dict[str, Any]
            Hyperparameters for PSO and solver behavior.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization.
        """
        try:
            super().__init__(scenario=scenario, rng=rng, parameters=parameters)
            self.name: str = "PSO"
            self.id: int = 2  # Solver identification
            self.controls_history: Optional[List[Tuple[float, float]]] = None
            self.n_control_points: int = int(parameters.get("n_control_points", 20))
            self.n_particles: int = int(parameters.get("n_particles", 100))
            self.print_enabled: bool = bool(parameters.get("print", False))

            self.max_steps: Optional[int] = None
            self.time_step: Optional[float] = None
            self.ship_velocity: Optional[float] = None
            self.goal: Optional[np.ndarray] = None
            self.start: Optional[np.ndarray] = None
            self.max_scenario_size: Optional[float] = None
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.__init__ failed: {e}")
            raise

    def log(self, msg: str) -> None:
        """
        Print a message if logging is enabled.

        Parameters
        ----------
        msg : str
            Message to print.
        """
        try:
            if self.print_enabled:
                print(msg)
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.log failed: {e}")
            raise

    # ---------------- control and helpers ----------------
    def control(self, step: int, state: List[float]) -> Tuple[float, float]:
        """
        Control law used during simulation: returns (ship_velocity, angular_rate) from
        the expanded control sequence if available; otherwise, straight motion.

        Parameters
        ----------
        step : int
            Current integration step.
        state : list[float]
            Current state [x, y, heading] (unused here).

        Returns
        -------
        tuple[float, float]
            (velocity, angular_rate)
        """
        try:
            if self.controls_history is None or not hasattr(self, "num_controls_history") or self.num_controls_history == 0:
                return (float(self.ship_velocity), 0.0)
            idx = min(step, self.num_controls_history - 1)
            return self.controls_history[idx]
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.control failed: {e}")
            raise

    def expandControlValues(self, control_values: np.ndarray | List[float]) -> np.ndarray:
        """
        Expand control points to a full-length horizon via linear interpolation.

        Parameters
        ----------
        control_values : np.ndarray | list[float]
            Control points (angular rates).

        Returns
        -------
        np.ndarray
            Interpolated sequence of length `max_steps`.
        """
        try:
            return interpolateControls(np.asarray(control_values, dtype=np.float64), int(self.max_steps))
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.expandControlValues failed: {e}")
            raise

    def makeControlFunctionFromU(self, u_full: np.ndarray) -> Callable[[int, List[float]], Tuple[float, float]]:
        """
        Create a step-indexed control function from a dense profile of angular rates.

        Parameters
        ----------
        u_full : np.ndarray
            Angular-rate sequence for all steps.

        Returns
        -------
        Callable[[int, list[float]], tuple[float, float]]
            Function mapping (step, state) -> (velocity, angular_rate).
        """
        try:
            v_ship = float(self.ship_velocity)

            def control_function(step: int, state: List[float]) -> Tuple[float, float]:
                k = int(step) if step is not None else 0
                k = max(0, min(k, len(u_full) - 1))
                return (v_ship, float(u_full[k]))

            return control_function
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.makeControlFunctionFromU failed: {e}")
            raise

    def heuristic(self, node, goal, add_turn_time: bool = True) -> float:
        """
        Admissible time-based heuristic to estimate remaining effort from a node to the goal.

        Parameters
        ----------
        node : tuple[float, float, float]
            (x, y, heading) of the current node.
        goal : tuple[float, float]
            (xg, yg) goal coordinates.
        add_turn_time : bool, optional
            If True, include a minimal turning-time estimate using r_max.

        Returns
        -------
        float
            Heuristic value (seconds).
        """
        try:
            x, y, theta = node
            xg, yg = goal
            r_max = self.scenario.getRMax()
            dx, dy = xg - x, yg - y
            d = self.distance((x, y), (xg, yg))
            if d < self.scenario.getGoalRadius():
                return 0.0
            ux, uy = dx / (d + 1e-6), dy / (d + 1e-6)
            cx, cy = self.scenario.getCurrentField().getCurrentAtPosition(x, y)
            c_dot_u = cx * ux + cy * uy
            v_s = self.scenario.getShipVelocity()
            v_parallel_best = v_s + c_dot_u
            max_time = float(self.max_steps) * float(self.time_step)
            base_time = max_time if v_parallel_best <= 0.0 else d / max(1e-1, v_parallel_best)
            if add_turn_time and r_max is not None and r_max > 0:
                beta = np.arctan2(uy, ux)
                delta = self.angleWrap(beta - theta)
                t_turn_min = abs(delta) / r_max
                base_time = t_turn_min + base_time
            return float(base_time)
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.heuristic failed: {e}")
            raise

    def integrateOnce(self, u_values) -> Tuple[bool, int, float, np.ndarray]:
        """
        Run a single forward simulation using a candidate angular-rate profile.

        Parameters
        ----------
        u_values : array-like
            Control points to be expanded over the horizon.

        Returns
        -------
        tuple
            (reached_goal, num_steps, total_distance, final_state)
        """
        try:
            u_arr = np.asarray(u_values, dtype=np.float64)
            u_full = self.expandControlValues(u_arr)
            control_fn = self.makeControlFunctionFromU(u_full)
            initial_state = self.scenario.getInitialState()
            sim = self.simulate(
                sim_id="PSO",
                state=initial_state,
                max_steps=self.max_steps,
                time_step=self.time_step,
                control_function=control_fn
            )
            if sim is None or sim.get("num_steps", 0) <= 5:
                fake_final = np.array(initial_state, dtype=float)
                return False, int(self.max_steps), 1e12, fake_final
            reached = bool(sim.get("goal_objective", False))
            num_steps = int(sim.get("num_steps", int(self.max_steps)))
            total_distance = float(sim.get("total_distance", 0.0))
            states_history = sim.get("states_history", [initial_state])
            final_state = np.array(states_history[-1], dtype=float)
            return reached, num_steps, total_distance, final_state
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.integrateOnce failed: {e}")
            raise

    def fitness(self, u_values) -> float:
        """
        Objective function for PSO/SLSQP: penalizes elapsed time, estimated remaining
        time (heuristic), and traveled distance scaled by ship speed.

        Parameters
        ----------
        u_values : array-like
            Candidate angular-rate control points.

        Returns
        -------
        float
            Fitness value to minimize.

        Raises
        ------
        Exception
            Propagates any error from simulation or heuristic evaluation.
        """
        try:
            if u_values is None or not np.all(np.isfinite(u_values)):
                print(f"[WARN] SolverZermeloPSO.fitness received invalid u_values: {u_values}")
                return float('inf')
            reached, num_steps, total_distance, final_state = self.integrateOnce(u_values)
            if final_state is None or not np.all(np.isfinite(final_state)):
                print(f"[DEBUG] integrateOnce returned invalid final_state for u_values: {u_values}")
                return float('inf')
            solution_time = float(num_steps) * float(self.time_step)
            v_ship = max(float(self.ship_velocity), 1e-9)
            node = (final_state[0], final_state[1], final_state[2] if len(final_state) >= 3 else 0.0)
            node_value = float(self.heuristic(node, tuple(self.goal), True))
            value = solution_time + node_value + total_distance / v_ship
            return float(value) if np.isfinite(value) else float('inf')
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.fitness failed: {e}")
            raise

    def generateTargetPoints(self, state, R_factor: float = 0.25) -> List[np.ndarray]:
        """
        Generate a small set of intermediate waypoints on an arc of radius R around the goal.

        Parameters
        ----------
        state : array-like
            Current state [x, y, heading]; only position is used.
        R_factor : float, optional
            Fraction of the distance-to-goal used as waypoint radius.

        Returns
        -------
        list[np.ndarray]
            Candidate intermediate waypoints (possibly empty).
        """
        try:
            goal = np.array(self.scenario.getGoal())
            state_pos = np.array(state[:2])
            r = float(self.scenario.getGoalRadius())
            d = np.linalg.norm(state_pos - goal)
            if d < 1e-9:
                return []
            R = d * R_factor
            if R < 2.0 * r:
                return []
            v = state_pos - goal
            dv = np.linalg.norm(v)
            if dv < R - 1e-12:
                return []
            base_angle = np.arctan2(v[1], v[0])
            val = np.clip(R / max(dv, 1.0e-12), -1.0, 1.0)
            alpha = np.arccos(val)
            arc_span = 2.0 * alpha
            if arc_span <= 1.0e-12:
                return [goal + R * np.array([np.cos(base_angle), np.sin(base_angle)])]
            ratio = np.clip(r / (2.0 * R), 0.0, 1.0)
            delta_max = 2.0 * np.arcsin(ratio)
            s_cover = 2.0 * delta_max
            if s_cover >= arc_span - 1.0e-12:
                center_angle = base_angle
                return [goal + R * np.array([np.cos(center_angle), np.sin(center_angle)])]
            first_angle = base_angle - alpha + delta_max
            last_angle = base_angle + alpha - delta_max
            inner_span = (2.0 * alpha) - 2.0 * delta_max
            n_intervals = int(np.ceil(inner_span / s_cover))
            n_centers = n_intervals + 1
            delta_angle = np.arctan2(np.sin(last_angle - first_angle), np.cos(last_angle - first_angle))
            center_angles = [first_angle + delta_angle * t for t in np.linspace(0.0, 1.0, n_centers)]
            return [goal + R * np.array([np.cos(a), np.sin(a)]) for a in center_angles] + [goal]
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO.generateTargetPoints failed: {e}")
            raise

    def _generateAnalyticGuess(self, intermediate_point: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Build an initial angular-rate profile by chaining analytic segments.

        Parameters
        ----------
        intermediate_point : np.ndarray | None
            Optional waypoint. If None, use a direct segment.

        Returns
        -------
        np.ndarray | None
            Clipped angular-rate control points (length = n_control_points), or None if generation fails.
        """
        try:
            waypoints = [intermediate_point] if intermediate_point is not None else []
            current_state = self.scenario.getInitialState()
            full_states_history = [current_state]
            for i, target_point in enumerate(waypoints):
                self.setTargetPoints([target_point])
                sim_data_segment = super().simulate(
                    sim_id=f"analytic_guess_segment_{i}",
                    state=current_state,
                    max_steps=self.max_steps,
                    time_step=self.time_step,
                    control_function=super().control
                )
                if (not sim_data_segment) or ("states_history" not in sim_data_segment) or \
                   (len(sim_data_segment["states_history"]) <= 1):
                    return None
                states_history = sim_data_segment["states_history"]
                full_states_history.extend(states_history[1:])
                current_state = states_history[-1]
            if len(full_states_history) < 2:
                return None
            thetas = np.array([s[2] for s in full_states_history], dtype=float)
            unwrapped_thetas = np.unwrap(thetas)
            u_values = np.diff(unwrapped_thetas) / float(self.time_step)
            if len(u_values) == 0:
                return None
            u_values = np.append(u_values, u_values[-1])
            indices = np.linspace(0, len(u_values) - 1, self.n_control_points, dtype=int)
            r_min, r_max = self.scenario.getRMin(), self.scenario.getRMax()
            return np.clip(u_values[indices], r_min, r_max)
        except Exception as e:
            print(f"[ERROR] SolverZermeloPSO._generateAnalyticGuess failed: {e}")
            raise

    def solve(self, max_steps, time_step, max_execution_time, seed: Optional[int] = None) -> Optional[dict[str, Any]]:
        """
        Run the full PSO pipeline:
          1) Set horizon and context.
          2) Generate analytic-seeded initial candidates.
          3) Run staged PSO with safe mutation and custom termination.
          4) Optionally refine with SLSQP.
          5) Simulate the best profile and return the result dictionary.

        Parameters
        ----------
        max_steps : int
            Maximum number of simulation steps (horizon).
        time_step : float
            Integration step size.
        max_execution_time : float
            Overall allowed wall-clock time for optimization.
        seed : int | None, optional
            Random seed for reproducibility. If None, derive from `rng`.

        Returns
        -------
        dict[str, Any] | None
            Final simulation result or None if no solution was produced.

        Raises
        ------
        Exception
            Propagates any optimization or simulation error.
        """
        try:
            self.max_steps = int(max_steps)
            self.time_step = float(time_step)
            self.ship_velocity = float(self.scenario.getShipVelocity())
            self.goal = np.array(self.scenario.getGoal(), dtype=float)
            self.start = np.array(self.scenario.getStart(), dtype=float)
            self.max_scenario_size = max(self.scenario.getSize())
            initial_state = self.scenario.getInitialState()
            execution_start_time = time.perf_counter()

            # Adapt hyperparameters to scenario scale
            if self.max_scenario_size < 1000:
                self.n_control_points, self.n_particles = 10, 50
            elif self.max_scenario_size < 5000:
                self.n_control_points, self.n_particles = 20, 100
            else:
                self.n_control_points, self.n_particles = 40, 150

            local_seed = int(seed) if seed is not None else int(self.rng.integers(0, 1_000_000_000))

            strategies_point_lists = [self.generateTargetPoints(initial_state, R_factor=0.25)]
            all_individual_points = [p for plist in strategies_point_lists for p in plist]
            all_individual_points.append(self.scenario.getGoal())

            initial_guesses: List[np.ndarray] = []
            for point in all_individual_points:
                guess = self._generateAnalyticGuess(point)
                if guess is not None:
                    initial_guesses.append(guess)
            if not initial_guesses:
                raise RuntimeError("Failed to generate initial candidate(s) for PSO.")

            best_u_pso = initial_guesses[0].copy()
            problem = self._ZermeloProblem(self)
            best_solutions_pool = initial_guesses.copy()

            # Multi-stage PSO with seeding from last bests
            for stage in range(3):
                initial_pop = self.rng.uniform(
                    low=float(self.scenario.getRMin()),
                    high=float(self.scenario.getRMax()),
                    size=(int(self.n_particles), int(self.n_control_points)),
                ).astype(np.float64)
                for i, u_seed in enumerate(best_solutions_pool):
                    if i < initial_pop.shape[0]:
                        initial_pop[i, :] = np.clip(u_seed, self.scenario.getRMin(), self.scenario.getRMax())
                remaining_time_stage = max(0.0, max_execution_time - (time.perf_counter() - execution_start_time))
                termination = TimeOrNoImprovement(
                    max_time=remaining_time_stage,
                    n_last=30,
                    tol=1e-6,
                    logger=self.log,
                )
                if not np.isfinite(initial_pop).all():
                    initial_pop = np.nan_to_num(initial_pop, nan=0.0, posinf=self.scenario.getRMax(),
                                                neginf=self.scenario.getRMin())

                res_pso = pymoo_minimize(
                    problem,
                    PSO(
                        pop_size=int(self.n_particles),
                        sampling=initial_pop,
                        w=0.729, c1=1.494, c2=1.494,
                        mutation=pm.PolynomialMutation(eta=20, prob=0.1),
                        use_cache=False
                    ),
                    termination,
                    seed=local_seed,
                    verbose=False
                )

                if res_pso.F is not None and len(res_pso.F) > 0:
                    try:
                        if self.fitness(best_u_pso) > float(res_pso.F[0]):
                            best_u_pso = np.asarray(res_pso.X, dtype=float)
                    except Exception:
                        # If fitness(best_u_pso) fails, accept the new candidate
                        best_u_pso = np.asarray(res_pso.X, dtype=float)
                    best_solutions_pool.append(np.asarray(res_pso.X, dtype=float))
                    best_solutions_pool = best_solutions_pool[-5:]

            # Local SLSQP refinement if time remains
            best_u = best_u_pso
            remaining_time = max_execution_time - (time.perf_counter() - execution_start_time)
            if remaining_time > 0.0:
                res_scipy = scipy_minimize(
                    self.fitness,
                    best_u_pso,
                    method="SLSQP",
                    bounds=[(self.scenario.getRMin(), self.scenario.getRMax())] * int(self.n_control_points),
                    options={"maxiter": int(self.max_steps * 4), "ftol": 1e-8},
                )
                if res_scipy.success and float(res_scipy.fun) < float(self.fitness(best_u_pso)):
                    best_u = np.asarray(res_scipy.x, dtype=float)

            reached, num_steps, total_distance, final_state = self.integrateOnce(best_u)
            u_full = self.expandControlValues(best_u)
            horizon = num_steps + 1 if reached else int(self.max_steps)
            self.controls_history = [(float(self.ship_velocity), float(u)) for u in u_full[:horizon]]
            self.num_controls_history = len(self.controls_history)
            sim_res = self.simulate(
                sim_id="PSO",
                state=self.scenario.getInitialState(),
                max_steps=horizon,
                time_step=self.time_step
            )
            return sim_res

        except Exception:
            self.log(f"[ERROR] SolverZermeloPSO.solve error:\n{traceback.format_exc()}")
            raise
