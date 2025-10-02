# ===============================================================================
# SolverZermeloIpopt Class (deterministic, single-mesh, direct)
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This class implements an optimal control solver for the Zermelo Navigation
#   Problem using Pyomo and the IPOPT optimizer. It constructs a single
#   Non-Linear Program (NLP) based on a direct method with Euler forward
#   discretization. The primary objective is to minimize total travel time.
#   The implementation is deterministic and does not use multi-grid or
#   warm-starting techniques for simplicity and robustness.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import time
import pyomo.environ as pyo
from problems.zermelo.solvers.SolverZermeloAnalytic import SolverZermeloAnalytic
from problems.zermelo.ScenarioZermelo import ScenarioZermelo

logging.getLogger("pyomo.core").setLevel(logging.ERROR)


class SolverZermeloIpopt(SolverZermeloAnalytic):
    """Ipopt-based solver for Zermelo's navigation problem.

    This solver formulates the problem as a single Non-Linear Program (NLP)
    using Pyomo. It minimizes total travel time with Euler forward
    discretization, enforces heading-rate bounds, and uses a goal-disk
    terminal constraint. It is designed as a single-mesh, direct solve
    without multi-grid refinement.
    """

    def __init__(self, scenario: ScenarioZermelo, rng: np.random.Generator, parameters: Dict) -> None:
        """Initialize the Ipopt solver instance.

        Parameters
        ----------
        scenario : ScenarioZermelo
            The scenario object containing environment and goal information.
        rng : np.random.Generator
            A random number generator (used for consistency in the solver hierarchy).
        parameters : dict
            A dictionary of configuration parameters for the solver.

        Raises
        ------
        Exception
            Propagates any error that occurs during initialization.
        """
        try:
            super().__init__(scenario=scenario, rng=rng, parameters=parameters)
            self.id: int = 3
            self.name: str = "Pyomo"
            self.controls_history: Optional[List[Tuple[float, float]]] = None

            p: Dict = parameters or {}
            self.ipopt_linear_solver: str = p.get("ipopt_linear_solver", "mumps")
            self.print_enabled: bool = bool(p.get("print", False))

            self.max_steps: Optional[int] = None
            self.time_step: Optional[float] = None
            self.start: Optional[Tuple[float, float]] = None
            self.goal: Optional[Tuple[float, float]] = None
            self.goal_radius: Optional[float] = None
            self.r_min: Optional[float] = None
            self.r_max: Optional[float] = None
            self.initial_heading: Optional[float] = None
            self.ship_velocity: Optional[float] = None
            self.current_velocity: Optional[float] = None
            self.scenario_size_x: Optional[float] = None
            self.scenario_size_y: Optional[float] = None
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt.__init__ failed: {e}")
            raise

    def _log(self, *a: Any) -> None:
        """Print log messages if printing is enabled.

        Parameters
        ----------
        *a : Any
            Variable length argument list to be printed.

        Returns
        -------
        None
        """
        try:
            if self.print_enabled:
                print(*a)
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt._log failed: {e}")
            raise

    def controlHistory(self, step: int, state: List[float]) -> Tuple[float, float]:
        """Return the control input at a given simulation step.

        This method acts as a callback for the simulator to replay the
        trajectory computed by the optimizer.

        Parameters
        ----------
        step : int
            The current simulation step index.
        state : List[float]
            The current state of the vessel (unused in this implementation).

        Returns
        -------
        Tuple[float, float]
            A tuple containing (ship_velocity, heading_rate).

        Raises
        ------
        Exception
            Propagates any error during control retrieval.
        """
        try:
            if self.controls_history is None:
                ship_velocity: float = self.scenario.getShipVelocity()
                return (ship_velocity, 0.0)
            if step < self.num_controls_history:
                return self.controls_history[step]
            return self.controls_history[self.num_controls_history - 1]
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt.controlHistory failed: {e}")
            raise

    def buildDiscreteModel(self, N: int) -> pyo.ConcreteModel:
        """Build the Pyomo NLP for a given number of discretization steps.

        Parameters
        ----------
        N : int
            The number of time steps (N+1 time points from 0 to N).

        Returns
        -------
        pyo.ConcreteModel
            The constructed Pyomo model instance.

        Raises
        ------
        Exception
            Propagates any error during model construction.
        """
        try:
            model: pyo.ConcreteModel = pyo.ConcreteModel()

            model.time = pyo.RangeSet(0, N)
            model.input_time = pyo.RangeSet(0, N - 1)

            model.x = pyo.Var(model.time, within=pyo.Reals, initialize=self.start[0])
            model.y = pyo.Var(model.time, within=pyo.Reals, initialize=self.start[1])
            model.theta = pyo.Var(model.time, within=pyo.Reals, initialize=self.initial_heading)
            model.u = pyo.Var(model.input_time, within=pyo.Reals, bounds=(self.r_min, self.r_max))

            d0: float = float(np.hypot(self.goal[0] - self.start[0], self.goal[1] - self.start[1]))
            v_eff: float = max(1e-6, self.ship_velocity + self.current_velocity)
            T0: float = max(0.1 * N * (self.time_step or 1.0), d0 / v_eff)
            model.T = pyo.Var(within=pyo.PositiveReals, initialize=T0)
            model.dt = pyo.Expression(rule=lambda m: m.T / N)

            model.x[0].fix(self.start[0])
            model.y[0].fix(self.start[1])
            model.theta[0].fix(self.initial_heading)

            model.final_x = model.x[N]
            model.final_y = model.y[N]
            model.final_goal_distance = pyo.Expression(
                expr=(model.final_x - self.goal[0]) ** 2.0 + (model.final_y - self.goal[1]) ** 2.0
            )
            model.final_goal_constraint = pyo.Constraint(
                expr=model.final_goal_distance <= (self.goal_radius ** 2.0)
            )

            def dyn_x(m, k):
                current_x, _ = self.scenario.getCurrentField().getCurrentAtPosition(m.x[k], m.y[k])
                return m.x[k + 1] == m.x[k] + m.dt * (self.ship_velocity * pyo.cos(m.theta[k]) + current_x)

            def dyn_y(m, k):
                _, current_y = self.scenario.getCurrentField().getCurrentAtPosition(m.x[k], m.y[k])
                return m.y[k + 1] == m.y[k] + m.dt * (self.ship_velocity * pyo.sin(m.theta[k]) + current_y)

            def dyn_theta(m, k):
                return m.theta[k + 1] == m.theta[k] + m.dt * m.u[k]

            model.ode_x = pyo.Constraint(model.input_time, rule=dyn_x)
            model.ode_y = pyo.Constraint(model.input_time, rule=dyn_y)
            model.ode_theta = pyo.Constraint(model.input_time, rule=dyn_theta)

            model.obj = pyo.Objective(expr=model.T + model.final_goal_distance / self.ship_velocity, sense=pyo.minimize)
            return model
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt.buildDiscreteModel failed: {e}")
            raise

    def _ipoptOptions(self, solver: pyo.SolverFactory) -> None:
        """Configure robust and generic options for the IPOPT solver.

        Parameters
        ----------
        solver : pyo.SolverFactory
            The Pyomo solver instance to configure.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during option configuration.
        """
        try:
            tolerance: float = 1e-6
            solver.options["tol"] = tolerance
            solver.options["constr_viol_tol"] = tolerance
            solver.options["acceptable_tol"] = 1e-4
            solver.options["acceptable_constr_viol_tol"] = 1e-4
            solver.options["mu_strategy"] = "adaptive"
            solver.options["hessian_approximation"] = "limited-memory"
            solver.options["nlp_scaling_method"] = "gradient-based"
            solver.options["linear_solver"] = self.ipopt_linear_solver
            solver.options["max_iter"] = 50000
            solver.options["print_level"] = 0
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt._ipoptOptions failed: {e}")
            raise

    def _solveOnce(self, N: int, time_budget: Optional[float]) -> Optional[Dict[str, Any]]:
        """Solve the NLP once for `N` steps without warm-starting.

        Parameters
        ----------
        N : int
            The number of discretization steps.
        time_budget : Optional[float]
            The maximum CPU time in seconds for the solver.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary with the solution results if successful, otherwise None.

        Raises
        ------
        Exception
            Propagates any error during the solve process.
        """
        try:
            model: pyo.ConcreteModel = self.buildDiscreteModel(N)
            solver: pyo.SolverFactory = pyo.SolverFactory("ipopt")
            self._ipoptOptions(solver)
            if time_budget is not None and time_budget > 0:
                solver.options["max_cpu_time"] = float(time_budget)

            results = solver.solve(model, tee=False)
            termination = results.solver.termination_condition

            if termination not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal, pyo.TerminationCondition.feasible]:
                self._log(f"⚠️ IPOPT failed at N={N} | term={termination}")
                return None

            T_star: float = float(pyo.value(model.T))
            dt_star: float = T_star / N
            u: np.ndarray = np.array([float(pyo.value(model.u[k])) for k in model.input_time], dtype=float)
            x: np.ndarray = np.array([float(pyo.value(model.x[k])) for k in model.time], dtype=float)
            y: np.ndarray = np.array([float(pyo.value(model.y[k])) for k in model.time], dtype=float)
            th: np.ndarray = np.array([float(pyo.value(model.theta[k])) for k in model.time], dtype=float)

            return {"N": N, "T": T_star, "dt": dt_star, "u": u, "x": x, "y": y, "theta": th}
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt._solveOnce failed: {e}")
            raise

    def solve(self, max_steps: int, time_step: float, max_execution_time: float) -> Optional[dict]:
        """Execute the main solver method.

        This method orchestrates the solution process, caching scenario parameters
        and invoking the single-mesh Pyomo/Ipopt solver.

        Parameters
        ----------
        max_steps : int
            The maximum number of simulation steps allowed.
        time_step : float
            The duration of a single simulation step.
        max_execution_time : float
            The total time budget for the solver in seconds.

        Returns
        -------
        Optional[dict]
            A dictionary with the best simulation results found, or None.

        Raises
        ------
        Exception
            Propagates any error during the solving process.
        """
        try:
            self.max_steps = int(max_steps)
            self.time_step = float(time_step)
            self.start = self.scenario.getStart()
            self.goal = self.scenario.getGoal()
            initial_state: np.ndarray = self.scenario.getInitialState()
            self.goal_radius = float(self.scenario.getGoalRadius())
            self.r_min = float(self.scenario.getRMin())
            self.r_max = float(self.scenario.getRMax())
            self.initial_heading = float(self.scenario.getInitialHeading())
            self.ship_velocity = float(self.scenario.getShipVelocity())
            self.current_velocity = float(self.scenario.getCurrentField().getVelocity())
            self.scenario_size_x = float(self.scenario.getSizeX())
            self.scenario_size_y = float(self.scenario.getSizeY())
            execution_start_time: float = time.perf_counter()
            best_ipopt_simulation_data: Optional[Dict] = None

            analytic_simulation_data: Optional[Dict] = super().solve(
                max_steps=max_steps, time_step=time_step, max_execution_time=max_execution_time
            )

            if analytic_simulation_data is not None and analytic_simulation_data['goal_objective']:
                best_ipopt_simulation_data = analytic_simulation_data
                best_ipopt_simulation_data['id'] = "Ipopt"
                self.max_steps = best_ipopt_simulation_data['num_steps']

            num_steps: int = self.max_steps
            while True:
                self._log(f"[IPOPT] solve with num_steps = {num_steps}")
                execution_time: float = time.perf_counter() - execution_start_time
                remaining_time: float = max_execution_time - execution_time
                if remaining_time < 0:
                    break
                self._log(f"[IPOPT] execution_time = {execution_time:.2f}")
                self._log(f"[IPOPT] remaining_time = {remaining_time:.2f}")

                solution: Optional[Dict] = self._solveOnce(N=num_steps, time_budget=remaining_time)
                if solution is None:
                    num_steps = num_steps + 1
                else:
                    num_steps = num_steps - 1
                    ipopt_simulation_data: Optional[Dict] = self.generateSimulationDataFromSolution(solution, initial_state)
                    if ipopt_simulation_data is not None:
                        if best_ipopt_simulation_data is None:
                            best_ipopt_simulation_data = ipopt_simulation_data
                        else:
                            best_ipopt_simulation_data = self.chooseBestSolution(
                                best_ipopt_simulation_data, ipopt_simulation_data
                            )
            return best_ipopt_simulation_data
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt.solve failed: {e}")
            raise

    def generateSimulationDataFromSolution(self, solution: Dict, initial_state: np.ndarray) -> Optional[Dict]:
        """Generate full simulation data from an Ipopt solution dictionary.

        This method reconstructs the control history and runs a final, precise
        simulation to get accurate trajectory and performance metrics.

        Parameters
        ----------
        solution : Dict
            The dictionary containing the results from `_solveOnce`.
        initial_state : np.ndarray
            The initial state vector for starting the simulation.

        Returns
        -------
        Optional[Dict]
            A dictionary with the full simulation data, or None if simulation fails.

        Raises
        ------
        Exception
            Propagates any error during simulation data generation.
        """
        try:
            T_star: float = float(solution["T"])
            dt_star: float = float(solution["dt"])
            u_star: np.ndarray = np.asarray(solution["u"], dtype=float)
            N: int = len(u_star)
            ship_velocity: float = self.scenario.getShipVelocity()
            x_star: np.ndarray = solution["x"]
            y_star: np.ndarray = solution["y"]
            last_position_astar: Tuple[float, float] = (x_star[N], y_star[N])
            distance_to_goal_astar: float = self.distance(last_position_astar, self.scenario.getGoal())
            self._log(f"[IPOPT] solution: T = {T_star:.3f} s, dt = {dt_star:.4f} s, N = {N}, last_position_astar = {last_position_astar}, distance_to_goal_astar={distance_to_goal_astar}")

            self.controls_history = [(ship_velocity, float(u_star[k])) for k in range(len(u_star))]
            self.num_controls_history = len(self.controls_history)

            margin_steps: int = max(3, int(0.05 * len(u_star)))
            num_steps_needed: int = len(u_star) + margin_steps
            self._log(f"[IPOPT] simulating {num_steps_needed} steps (N={N} + margin={margin_steps})")
            
            ipopt_simulation_data: Optional[Dict] = self.simulate(
                sim_id="Ipopt",
                state=initial_state,
                max_steps=num_steps_needed,
                time_step=dt_star,
                control_function=self.controlHistory
            )
            if ipopt_simulation_data is not None:
                ipopt_simulation_data["time_step"] = dt_star
            return ipopt_simulation_data
        except Exception as e:
            print(f"[ERROR] SolverZermeloIpopt.generateSimulationDataFromSolution failed: {e}")
            raise