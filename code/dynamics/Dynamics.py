# ===============================================================================
# Dynamics Base Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.1
# Description:
#   Generic framework for simulating continuous-time dynamic systems with selectable
#   numerical integration schemes (Euler, Trapezoidal/Heun, and RK4). The class
#   defines a typed API for state-derivative evaluation, single-step integration,
#   and multi-step simulation with optional control/disturbance providers, stop
#   criteria, and wall-clock time limits. Subclasses must override `derivatives`.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import time
import numpy as np
from typing import Callable, Optional, Tuple, Any
from utils.Geometry import Geometry


class Dynamics(Geometry):
    """
    Base class for dynamic-systems simulation with pluggable integrators.

    Attributes
    ----------
    parameters : dict[str, Any]
        Configuration parameters (e.g., {'integration': 'euler'|'trapezoidal'|'rk4'}).
    num_states : int
        Number of state variables.
    num_controls : int
        Number of control input variables.
    num_disturbances : int
        Number of disturbance variables.
    """

    def __init__(self, num_states: int, num_controls: int, num_disturbances: int, parameters: dict[str, Any]) -> None:
        """
        Initialize the dynamics system with state/control/disturbance dimensions and parameters.

        Parameters
        ----------
        num_states : int
            Number of state variables.
        num_controls : int
            Number of control inputs.
        num_disturbances : int
            Number of disturbance components.
        parameters : dict[str, Any]
            Configuration dictionary; must include the 'integration' key with one of:
            'euler', 'trapezoidal', or 'rk4'. If missing, subclasses may set it.

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
            self.parameters: dict[str, Any] = parameters
            self.num_states: int = num_states
            self.num_controls: int = num_controls
            self.num_disturbances: int = num_disturbances
        except Exception as e:
            print(f"[ERROR] Dynamics.__init__ failed: {e}")
            raise

    def derivatives(self, state: np.ndarray, control: Any, disturbance: Optional[Any] = None) -> np.ndarray:
        """
        Compute the time derivatives of the system state vector.

        Parameters
        ----------
        state : np.ndarray
            Current state vector with length `num_states`.
        control : Any
            Control input object or array matching the system's control interface.
        disturbance : Any, optional
            Disturbance input object or array; None if not used.

        Returns
        -------
        np.ndarray
            Derivative vector (dx/dt) of shape (num_states,).

        Raises
        ------
        Exception
            Propagates any error during derivative computation.

        Notes
        -----
        Subclasses must override this method to implement system-specific dynamics.
        """
        try:
            return np.zeros(self.num_states)
        except Exception as e:
            print(f"[ERROR] Dynamics.derivatives failed: {e}")
            raise

    def stepEuler(
        self, time_step: float, state: np.ndarray, control: Any, disturbance: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one step using the explicit Euler method.

        Parameters
        ----------
        time_step : float
            Integration time step (seconds).
        state : np.ndarray
            Current state vector.
        control : Any
            Control input for this step.
        disturbance : Any, optional
            Disturbance input for this step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (next_state, state_derivatives) after one Euler step.

        Raises
        ------
        Exception
            Propagates any error during integration.
        """
        try:
            state_derivatives: np.ndarray = self.derivatives(state, control, disturbance)
            new_state: np.ndarray = state + time_step * state_derivatives
            return new_state, state_derivatives
        except Exception as e:
            print(f"[ERROR] Dynamics.stepEuler failed: {e}")
            raise

    def stepTrapezoidal(
        self, time_step: float, state: np.ndarray, control: Any, disturbance: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one step using the explicit Trapezoidal (Heun's) method.

        Parameters
        ----------
        time_step : float
            Integration time step (seconds).
        state : np.ndarray
            Current state vector.
        control : Any
            Control input for this step.
        disturbance : Any, optional
            Disturbance input for this step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (next_state, averaged_derivatives) after one Trapezoidal step.

        Raises
        ------
        Exception
            Propagates any error during integration.
        """
        try:
            k1: np.ndarray = self.derivatives(state, control, disturbance)
            k2: np.ndarray = self.derivatives(state + time_step * k1, control, disturbance)
            new_state: np.ndarray = state + (time_step / 2.0) * (k1 + k2)
            return new_state, 0.5 * (k1 + k2)
        except Exception as e:
            print(f"[ERROR] Dynamics.stepTrapezoidal failed: {e}")
            raise

    def stepRungeKutta(
        self, time_step: float, state: np.ndarray, control: Any, disturbance: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one step using the classical 4th-order Runge–Kutta (RK4) method.

        Parameters
        ----------
        time_step : float
            Integration time step (seconds).
        state : np.ndarray
            Current state vector.
        control : Any
            Control input for this step.
        disturbance : Any, optional
            Disturbance input for this step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (next_state, weighted_derivatives) after one RK4 step.

        Raises
        ------
        Exception
            Propagates any error during integration.
        """
        try:
            k1: np.ndarray = self.derivatives(state, control, disturbance)
            k2: np.ndarray = self.derivatives(state + 0.5 * time_step * k1, control, disturbance)
            k3: np.ndarray = self.derivatives(state + 0.5 * time_step * k2, control, disturbance)
            k4: np.ndarray = self.derivatives(state + time_step * k3, control, disturbance)
            new_state: np.ndarray = state + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return new_state, (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        except Exception as e:
            print(f"[ERROR] Dynamics.stepRungeKutta failed: {e}")
            raise

    def step(
        self,
        step: int,
        time_step: float,
        current_state: np.ndarray,
        integration_step_func: Optional[Callable[[float, np.ndarray, Any, Optional[Any]], Tuple[np.ndarray, np.ndarray]]] = None,
        control_function: Optional[Callable[[int, np.ndarray], Any]] = None,
        disturbance_function: Optional[Callable[[int, np.ndarray], Any]] = None,
        stop_function: Optional[Callable[[int, np.ndarray], bool]] = None,
    ) -> Tuple[np.ndarray, Any, Any, np.ndarray, bool]:
        """
        Perform a single simulation cycle: compute control/disturbance, integrate one step,
        and check the stop condition.

        Parameters
        ----------
        step : int
            Current step index (0-based).
        time_step : float
            Integration time step (seconds).
        current_state : np.ndarray
            Current state vector.
        integration_step_func : callable, optional
            One-step integrator with signature (dt, state, control, disturbance) -> (next_state, k).
        control_function : callable, optional
            Callable with signature (step, state) -> control; if None, no control is applied.
        disturbance_function : callable, optional
            Callable with signature (step, state) -> disturbance; if None, no disturbance is applied.
        stop_function : callable, optional
            Callable with signature (step, state) -> bool to request early termination.

        Returns
        -------
        tuple
            (next_state, control, disturbance, state_derivatives, stop_simulation).

        Raises
        ------
        Exception
            Propagates any error during step evaluation or integration.
        """
        try:
            if control_function is not None:
                if callable(control_function):
                    control: Any = control_function(step, current_state)
                else:
                    control = control_function
            else:
                control = None

            disturbance: Any = disturbance_function(step, current_state) if disturbance_function else None
            stop_simulation: bool = stop_function(step, current_state) if stop_function else False

            next_state, state_derivatives = integration_step_func(time_step, current_state, control, disturbance)
            return next_state, control, disturbance, state_derivatives, stop_simulation
        except Exception as e:
            print(f"[ERROR] Dynamics.step failed: {e}")
            raise

    def simulate(
        self,
        state: np.ndarray,
        max_steps: int,
        time_step: float,
        max_execution_time: Optional[float] = None,
        control_function: Optional[Callable[[int, np.ndarray], Any]] = None,
        disturbance_function: Optional[Callable[[int, np.ndarray], Any]] = None,
        stop_function: Optional[Callable[[int, np.ndarray], bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Simulate the system for up to `max_steps`, honoring optional stop criteria and
        wall-clock limits.

        Parameters
        ----------
        state : np.ndarray
            Initial state vector of shape (num_states,).
        max_steps : int
            Maximum number of integration steps.
        time_step : float
            Integration time step (seconds).
        max_execution_time : float, optional
            Wall-clock time budget in seconds; if exceeded, the loop terminates early.
        control_function : callable, optional
            Provider of controls per step; see `step`.
        disturbance_function : callable, optional
            Provider of disturbances per step; see `step`.
        stop_function : callable, optional
            Predicate to request early termination; see `step`.

        Returns
        -------
        tuple
            (time_history, states_history, controls_history, disturbance_history, state_derivatives_history)
            where:
              - time_history : np.ndarray of shape (n,)
              - states_history : np.ndarray of shape (n, num_states)
              - controls_history : np.ndarray | None of shape (n, num_controls)
              - disturbance_history : np.ndarray | None of shape (n, num_disturbances)
              - state_derivatives_history : np.ndarray of shape (n, num_states)

        Raises
        ------
        Exception
            Propagates any error during the simulation loop or array slicing.
        """
        try:
            integration_method: str = str(self.parameters.get("integration", "euler"))
            if integration_method == "euler":
                integration_step_func = self.stepEuler
            elif integration_method == "trapezoidal":
                integration_step_func = self.stepTrapezoidal
            elif integration_method == "rk4":
                integration_step_func = self.stepRungeKutta
            else:
                raise ValueError(f"Unknown integration method: {integration_method}")

            time_history: np.ndarray = np.zeros(max_steps, dtype=float)
            states_history: np.ndarray = np.zeros((max_steps, self.num_states), dtype=float)
            state_derivatives_history: np.ndarray = np.zeros((max_steps, self.num_states), dtype=float)
            controls_history: Optional[np.ndarray] = (
                np.zeros((max_steps, self.num_controls), dtype=float) if control_function is not None else None
            )
            disturbance_history: Optional[np.ndarray] = (
                np.zeros((max_steps, self.num_disturbances), dtype=float) if disturbance_function is not None else None
            )

            current_state: np.ndarray = np.array(state, dtype=float)
            simulation_time: float = 0.0

            if max_execution_time is not None:
                execution_start_time: float = time.perf_counter()

            for step_idx in range(max_steps):
                time_history[step_idx] = simulation_time
                states_history[step_idx] = current_state

                next_state, control, disturbance, state_derivatives, stop_simulation = self.step(
                    step=step_idx,
                    time_step=time_step,
                    current_state=current_state,
                    integration_step_func=integration_step_func,
                    control_function=control_function,
                    disturbance_function=disturbance_function,
                    stop_function=stop_function,
                )

                if controls_history is not None:
                    controls_history[step_idx] = control
                if disturbance_history is not None:
                    disturbance_history[step_idx] = disturbance
                state_derivatives_history[step_idx] = state_derivatives

                if stop_simulation:
                    step_idx += 1
                    break

                if max_execution_time is not None:
                    if (time.perf_counter() - execution_start_time) >= max_execution_time:
                        step_idx += 1
                        break

                current_state = next_state
                simulation_time += time_step

            time_history = time_history[:step_idx]
            states_history = states_history[:step_idx]
            state_derivatives_history = state_derivatives_history[:step_idx]
            if controls_history is not None:
                controls_history = controls_history[:step_idx]
            if disturbance_history is not None:
                disturbance_history = disturbance_history[:step_idx]

            return time_history, states_history, controls_history, disturbance_history, state_derivatives_history
        except Exception as e:
            print(f"[ERROR] Dynamics.simulate failed: {e}")
            raise
