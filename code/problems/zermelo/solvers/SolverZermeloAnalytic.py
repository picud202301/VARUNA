# ===============================================================================
# SolverZermeloAnalytic — analytic controller for Zermelo Navigation Problem
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Implements an analytic proportional controller for Zermelo's navigation
#   problem. The solver computes steering commands to drive the vessel towards
#   sequential target points while compensating for environmental currents.
#
#   Key responsibilities:
#     - Control law using proportional feedback on angular error
#     - Integration of vessel dynamics via inherited simulation framework
#     - Support for bounded angular rate within scenario constraints
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Dict, List, Tuple, Optional
import numpy as np
from problems.zermelo.solvers.SolverZermelo import SolverZermelo
from problems.zermelo.ScenarioZermelo import ScenarioZermelo


class SolverZermeloAnalytic(SolverZermelo):
    """Analytic solver for the Zermelo navigation problem.

    This class uses a proportional control law to align the vessel's
    heading with the direction to a target, actively compensating for
    the effects of environmental currents.
    """

    def __init__(self, scenario: ScenarioZermelo, rng: np.random.Generator, parameters: Dict) -> None:
        """Initialize an analytic solver instance.

        Parameters
        ----------
        scenario : ScenarioZermelo
            The scenario object containing environment and goal information.
        rng : np.random.Generator
            A random number generator for any stochastic behavior.
        parameters : dict
            A dictionary of configuration parameters for the solver.

        Raises
        ------
        Exception
            Propagates any error that occurs during initialization.
        """
        try:
            super().__init__(scenario, rng, parameters)
            self.kp: float = 1.0
            self.id: int = 0  # Solver identification
        except Exception as e:
            print(f"[ERROR] SolverZermeloAnalytic.__init__ failed: {e}")
            raise

    def control(self, step: int, state: List[float]) -> np.ndarray:
        """Compute the control input using proportional feedback.

        This method calculates the required vessel velocity and angular rate
        to navigate towards the current target point.

        Parameters
        ----------
        step : int
            The current simulation step index.
        state : List[float]
            The current state of the vessel, represented as [x, y, heading].

        Returns
        -------
        np.ndarray
            A control vector containing [velocity, angular_rate].

        Raises
        ------
        Exception
            Propagates any error that occurs during control calculation.
        """
        try:
            x: float
            y: float
            theta: float
            x, y, theta = state

            # Current disturbance at vessel position
            current_x: float
            current_y: float
            current_x, current_y = self.scenario.getCurrentField().getCurrentAtPosition(x, y)

            # Target point (updates if reached)
            target_point: Tuple[float, float] = self.target_points[self.target_point_id]
            distance_to_target_point: float = self.distance((x, y), target_point)

            if distance_to_target_point <= self.scenario.getGoalRadius():
                self.target_point_id += 1
                self.target_point_id = min(self.target_point_id, len(self.target_points) - 1)
                target_point = self.target_points[self.target_point_id]

            dx_goal: float = target_point[0] - x
            dy_goal: float = target_point[1] - y

            # Desired heading towards target
            theta_desired: float = np.arctan2(dy_goal, dx_goal)

            # Ship velocity components
            ship_velocity: float = self.scenario.getShipVelocity()
            vx_ship: float = ship_velocity * np.cos(theta)
            vy_ship: float = ship_velocity * np.sin(theta)

            # Effective velocity with current disturbance
            vx_rel: float = vx_ship + current_x
            vy_rel: float = vy_ship + current_y
            theta_rel: float = np.arctan2(vy_rel, vx_rel)

            # Angular error in [-pi, pi]
            angle_error: float = np.arctan2(np.sin(theta_desired - theta_rel), np.cos(theta_desired - theta_rel))

            # Proportional angular velocity control
            r: float = self.kp * angle_error

            # Apply scenario limits
            r = np.clip(r, self.scenario.getRMin(), self.scenario.getRMax())

            return np.array([ship_velocity, r], dtype=float)

        except Exception as e:
            print(f"[ERROR] SolverZermeloAnalytic.control failed: {e}")
            raise

    def solve(self, max_steps: int, time_step: float, max_execution_time: float) -> Optional[Dict]:
        """Solve the Zermelo problem using the analytic control law.

        This method runs a full simulation from the initial state to the goal
        or until a termination condition is met.

        Parameters
        ----------
        max_steps : int
            The maximum number of simulation steps to execute.
        time_step : float
            The time increment for each step of the numerical integration.
        max_execution_time : float
            The maximum wall-clock time allowed for the simulation, in seconds.

        Returns
        -------
        Optional[dict]
            A dictionary with simulation results, including trajectories and
            performance metrics, or None if no solution was found.

        Raises
        ------
        Exception
            Propagates any error that occurs during the simulation.
        """
        try:
            # Proportional gain scaled with time step
            self.kp = 0.5 / time_step
            initial_state: np.ndarray = self.scenario.getInitialState()
            self.setTargetPoints(target_points=None)
            analytic_simulation_data: Optional[Dict] = self.simulate(
                sim_id="Analytic",
                state=initial_state,
                max_steps=max_steps,
                time_step=time_step,
                max_execution_time=max_execution_time
            )
            return analytic_simulation_data
        except Exception as e:
            print(f"[ERROR] SolverZermeloAnalytic.solve failed: {e}")
            raise