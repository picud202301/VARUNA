# ===============================================================================
# Zermelo Navigation Problem base solver
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Base class for solving Zermelo's navigation problem. It provides a framework
#   to integrate vessel dynamics, query ambient currents from the scenario, run
#   time-marched simulations, and evaluate resulting trajectories using a
#   navigation index that blends elapsed time and traveled distance.
#   Responsibilities:
#     - Manage scenario and solver parameters.
#     - Run simulations via DynamicsZermelo.
#     - Evaluate solutions by goal achievement and navigation index.
#     - Provide default control, disturbance, and stopping policies.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Callable
import numpy as np
from utils.Geometry import Geometry
from problems.zermelo.DynamicsZermelo import DynamicsZermelo


class SolverZermelo(Geometry):
    """
    Base solver class for Zermelo's navigation problem. It orchestrates the
    integration of vessel dynamics under environmental currents, manages solver
    state, and evaluates trajectories against a goal condition.

    Attributes
    ----------
    id : int
        Identifier for the solver instance or the most recent simulation series.
    parameters : dict[str, Any]
        Configuration parameters for the solver (e.g., integration method).
    rng : np.random.Generator
        Pseudo-random number generator injected for stochastic behaviors.
    scenario : object
        Scenario object exposing geometry, goal, ship speed, and current field.
    target_points : list[tuple[float, float]] | None
        Optional list of intermediate targets; the scenario goal is always appended.
    target_point_id : int
        Index of the current target within `target_points`.
    """

    def __init__(self, scenario: object, rng: np.random.Generator, parameters: dict[str, Any]) -> None:
        """
        Initialize a :class:`SolverZermelo` instance.

        Parameters
        ----------
        scenario : object
            Scenario object containing environment and goal information.
        rng : np.random.Generator
            Random number generator for stochastic behavior.
        parameters : dict[str, Any]
            Solver configuration parameters (e.g., integration method).

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization.
        """
        try:
            super().__init__()
            self.id: int = 0
            self.parameters: dict[str, Any] = parameters
            self.rng: np.random.Generator = rng
            self.scenario: object = scenario
            self.target_points: list[tuple[float, float]] | None = None
            self.target_point_id: int = 0
        except Exception as e:
            print(f"[ERROR] SolverZermelo.__init__ failed: {e}")
            raise

    def setTargetPoints(self, target_points: list[tuple[float, float]] | None = None) -> bool:
        """
        Set a list of target points for the solver (last point is the goal).

        Parameters
        ----------
        target_points : list[tuple[float, float]] | None, optional
            List of intermediate target coordinates (x, y). The final target is
            always the scenario goal.

        Returns
        -------
        bool
            ``True`` if the target points were successfully set.

        Raises
        ------
        Exception
            Propagates any error during list construction.
        """
        try:
            self.target_point_id = 0
            self.target_points = []
            if target_points is not None:
                self.target_points.extend(target_points)
            self.target_points.append(tuple(self.scenario.getGoal()))
            return True
        except Exception as e:
            print(f"[ERROR] SolverZermelo.setTargetPoints failed: {e}")
            raise

    def getNavegationIndex(self, simulation_data: dict[str, Any]) -> float:
        """
        Compute a navigation index combining elapsed time and traveled distance.

        Notes
        -----
        The index is defined as:
            ``total_time + total_distance / ship_velocity``
        Lower values indicate better performance.

        Parameters
        ----------
        simulation_data : dict[str, Any]
            Simulation results containing ``total_time``, ``total_distance``, and
            a scenario-level ship velocity.

        Returns
        -------
        float
            Navigation index (lower is better).

        Raises
        ------
        Exception
            Propagates any error if required fields are missing.
        """
        try:
            return float(simulation_data["total_time"]) + float(simulation_data["total_distance"]) / float(self.scenario.getShipVelocity())
        except Exception as e:
            print(f"[ERROR] SolverZermelo.getNavegationIndex failed: {e}")
            raise

    def chooseBestSolution(self, sim1: dict[str, Any], sim2: dict[str, Any]) -> dict[str, Any]:
        """
        Compare two simulation results and choose the best one.

        Decision rule
        -------------
        1) Prefer solutions that achieve the goal.
        2) If both achieve (or both fail), prefer the one with the lower navigation index.

        Parameters
        ----------
        sim1 : dict[str, Any]
            First simulation result.
        sim2 : dict[str, Any]
            Second simulation result.

        Returns
        -------
        dict[str, Any]
            The best solution according to the rule.

        Raises
        ------
        Exception
            Propagates any error during comparison.
        """
        try:
            if sim1.get("goal_objective", False) and not sim2.get("goal_objective", False):
                return sim1
            if sim2.get("goal_objective", False) and not sim1.get("goal_objective", False):
                return sim2
            index_sim1 = self.getNavegationIndex(sim1)
            index_sim2 = self.getNavegationIndex(sim2)
            return sim1 if index_sim1 < index_sim2 else sim2
        except Exception as e:
            print(f"[ERROR] SolverZermelo.chooseBestSolution failed: {e}")
            raise

    def simulate(
        self,
        sim_id: int,
        state: list[float],
        max_steps: int,
        time_step: float,
        max_execution_time: float | None = None,
        control_function: Callable[[int, list[float]], np.ndarray] | None = None,
        disturbance_function: Callable[[int, list[float]], np.ndarray] | None = None,
        stop_function: Callable[[int, list[float]], bool] | None = None,
    ) -> dict[str, Any] | None:
        """
        Run a simulation using :class:`DynamicsZermelo`.

        Parameters
        ----------
        sim_id : int
            Identifier for the simulation run.
        state : list[float]
            Initial state ``[x, y, heading]``.
        max_steps : int
            Maximum number of simulation steps.
        time_step : float
            Time step for integration.
        max_execution_time : float | None, optional
            Maximum wall-clock time allowed for the integration routine (seconds).
        control_function : Callable[[int, list[float]], np.ndarray] | None, optional
            Function providing control inputs per step, returning ``[v, r]``.
        disturbance_function : Callable[[int, list[float]], np.ndarray] | None, optional
            Function providing disturbances per step, returning current vector ``[u, v]``.
        stop_function : Callable[[int, list[float]], bool] | None, optional
            Function indicating whether to stop the simulation at a given step/state.

        Returns
        -------
        dict[str, Any] | None
            Simulation data including histories, metrics, and flags, or ``None``
            if no simulation was executed.

        Raises
        ------
        Exception
            Propagates any error raised by the dynamics integrator or post-processing.
        """
        try:
            dynamics_zermelo = DynamicsZermelo(
                num_states=3,
                num_controls=2,
                num_disturbances=2,
                parameters={"integration": self.parameters.get("integration", "euler")},
            )
            control_function = control_function or self.control
            disturbance_function = disturbance_function or self.disturbance
            stop_function = stop_function or self.stopSimulation

            (
                time_history,
                states_history,
                controls_history,
                disturbance_history,
                state_derivatives_history,
            ) = dynamics_zermelo.simulate(
                state=state,
                max_steps=max_steps,
                time_step=time_step,
                max_execution_time=max_execution_time,
                control_function=control_function,
                disturbance_function=disturbance_function,
                stop_function=stop_function,
            )

            num_steps: int = len(time_history)
            simulation_data: dict[str, Any] | None = None
            if num_steps > 0:
                total_time: float = time_step * num_steps
                last_state = states_history[-1]
                goal = np.array(self.scenario.getGoal())
                goal_radius = float(self.scenario.getGoalRadius())
                distance_to_goal: float = self.distance(np.array(last_state[:2]), goal)
                goal_objective: bool = bool(distance_to_goal <= goal_radius)
                total_distance: float = self.calculateTrajectoryDistance(states_history)  # keep original API

                simulation_data = {
                    "id": sim_id,
                    "time_step": time_step,
                    "num_steps": num_steps,
                    "goal_objective": goal_objective,
                    "total_time": total_time,
                    "total_distance": total_distance,
                    "last_state": last_state,  # np.ndarray
                    "distance_to_goal": distance_to_goal,
                    "time_history": time_history,  # np.ndarray
                    "states_history": states_history,  # np.ndarray
                    "controls_history": controls_history,  # np.ndarray
                    "disturbance_history": disturbance_history,  # np.ndarray
                    "state_derivatives_history": state_derivatives_history,  # np.ndarray
                }
                simulation_data["navegation_index"] = self.getNavegationIndex(simulation_data)

            return simulation_data
        except Exception as e:
            print(f"[ERROR] SolverZermelo.simulate failed: {e}")
            raise

    def summarySimulationData(self, simulation_data: dict[str, Any] | None) -> None:
        """
        Print a concise summary of simulation results.

        Parameters
        ----------
        simulation_data : dict[str, Any] | None
            Simulation data dictionary containing histories, metrics, and flags.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during formatted printing.
        """
        try:
            if simulation_data is None:
                print("No solution found!!")
                return

            print(
                f'---------------------- Simulation Data Summary : id : {simulation_data["id"]} ----------------------'
            )
            if "grid_time_factor" not in simulation_data:
                print(
                    f'Time : {simulation_data["total_time"]}, Steps: {simulation_data["num_steps"]}, '
                    f'Step time: {simulation_data["time_step"]}'
                )
            else:
                print(
                    f'Time : {simulation_data["total_time"]}, Steps: {simulation_data["num_steps"]}, '
                    f'Step time: {simulation_data["time_step"]}, Grid Time Factor: {simulation_data["grid_time_factor"]}'
                )
            print(f'Distance : {simulation_data["total_distance"]}')
            print(f'Last state : {simulation_data["last_state"]}')
            print(
                f'Distance to target: {simulation_data["distance_to_goal"]}, Goal Objective : '
                f'{simulation_data["goal_objective"]}'
            )
            if "execution_time" in simulation_data:
                print(f'Execution time : {simulation_data["execution_time"]}')
            if "astar_grid_time" in simulation_data:
                print(
                    f'Astar grid time : {simulation_data["astar_grid_time"]}  '
                    f'num steps: {simulation_data["astar_num_steps"]}'
                )
            print("----------------------------------------------------------------------------------------------------------------")
        except Exception as e:
            print(f"[ERROR] SolverZermelo.summarySimulationData failed: {e}")
            raise

    def stopSimulation(self, step: int, state: list[float]) -> bool:
        """
        Check the stopping condition for the simulation (goal reached).

        Parameters
        ----------
        step : int
            Current step number.
        state : list[float]
            Current state ``[x, y, heading]``.

        Returns
        -------
        bool
            ``True`` if the vessel reached the goal region; ``False`` otherwise.

        Raises
        ------
        Exception
            Propagates any error during goal check.
        """
        try:
            return self.scenario.isGoalReached(x=state[0], y=state[1])
        except Exception as e:
            print(f"[ERROR] SolverZermelo.stopSimulation failed: {e}")
            raise

    def control(self, step: int, state: list[float]) -> np.ndarray:
        """
        Default control policy placeholder: returns zero surge and zero turn-rate.

        Parameters
        ----------
        step : int
            Current simulation step.
        state : list[float]
            Current state ``[x, y, heading]``.

        Returns
        -------
        np.ndarray
            Control vector ``[velocity, angular_rate]``.

        Raises
        ------
        Exception
            Propagates any error during vector construction.
        """
        try:
            return np.array([0.0, 0.0], dtype=float)
        except Exception as e:
            print(f"[ERROR] SolverZermeloAnalytic.control failed: {e}")
            raise

    def disturbance(self, step: int, state: list[float]) -> np.ndarray:
        """
        Return the current disturbance affecting the vessel at its current position.

        Parameters
        ----------
        step : int
            Current step number.
        state : list[float]
            Current state ``[x, y, heading]``.

        Returns
        -------
        np.ndarray
            Current disturbance vector ``[u, v]`` at the vessel's position.

        Raises
        ------
        Exception
            Propagates any error from the scenario's current-field query.
        """
        try:
            return np.array(self.scenario.getCurrentField().getCurrentAtPosition(state[0], state[1]), dtype=float)
        except Exception as e:
            print(f"[ERROR] SolverZermelo.disturbance failed: {e}")
            raise

    def plot(self, plotter: object) -> None:
        """
        Optional hook to allow solver-specific overlays through a plotter.

        Parameters
        ----------
        plotter : object
            Plotting utility instance.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error raised during plotting.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] SolverZermelo.plot failed: {e}")
            raise

    def getScenario(self) -> object:
        """
        Return the scenario associated with this solver.

        Returns
        -------
        object
            The scenario instance.

        Raises
        ------
        Exception
            Propagates any error during attribute access.
        """
        try:
            return self.scenario
        except Exception as e:
            print(f"[ERROR] SolverZermelo.getScenario failed: {e}")
            raise

    def getCurrentField(self) -> object:
        """
        Return the current field associated with the scenario.

        Returns
        -------
        object
            The current-field provider.

        Raises
        ------
        Exception
            Propagates any error during provider access.
        """
        try:
            return self.scenario.getCurrentField()
        except Exception as e:
            print(f"[ERROR] SolverZermelo.getCurrentField failed: {e}")
            raise
