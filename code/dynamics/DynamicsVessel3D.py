# ===============================================================================
# DynamicsVessel3D Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Dynamics model for Zermelo's navigation problem, compatible with NumPy.
#   The model computes the time derivatives of the state given vessel controls
#   and environmental disturbances (currents).
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Optional, Dict
import numpy as np
from dynamics.Dynamics import Dynamics


class DynamicsVessel3D(Dynamics):
    """
    Dynamics model for Zermelo's navigation problem.

    State representation
    --------------------
    state : np.ndarray
        [x, y, theta] (position x, position y, heading angle [rad])

    Control input
    -------------
    control : np.ndarray
        [v, r] (speed [m/s], heading rate [rad/s])

    Disturbance input
    -----------------
    disturbance : np.ndarray
        [current_x, current_y] (water current components [m/s])
    """

    def __init__(self, num_states: int, num_controls: int, num_disturbances: int, parameters: Dict) -> None:
        """Initialize the 3D dynamics model.

        Parameters
        ----------
        num_states : int
            Number of system states (expected 3).
        num_controls : int
            Number of control inputs (expected 2).
        num_disturbances : int
            Number of disturbance variables (expected 2).
        parameters : dict
            Configuration parameters (e.g., integration method).

        Raises
        ------
        Exception
            If initialization fails.
        """
        try:
            super().__init__(num_states, num_controls, num_disturbances, parameters)
        except Exception as e:
            print(f"[ERROR] DynamicsVessel3D.__init__ failed: {e}")
            raise

    def derivatives(
        self, state: np.ndarray, control: np.ndarray, disturbance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute state derivatives.

        Parameters
        ----------
        state : np.ndarray
            State vector [x, y, theta].
        control : np.ndarray
            Control input [v, r].
        disturbance : np.ndarray, optional
            Disturbance vector [current_x, current_y]. If None, assumes no current.

        Returns
        -------
        np.ndarray
            Derivative vector [dx, dy, dtheta].

        Raises
        ------
        Exception
            If computation fails.
        """
        try:
            theta: float = float(state[2])
            v: float = float(control[0])
            r: float = float(control[1])
            current_x: float
            current_y: float

            if disturbance is None:
                current_x, current_y = 0.0, 0.0
            else:
                current_x = float(disturbance[0])
                current_y = float(disturbance[1])

            dx: float = v * np.cos(theta) + current_x
            dy: float = v * np.sin(theta) + current_y
            dtheta: float = r

            return np.array((dx, dy, dtheta), dtype=np.float64)
        except Exception as e:
            print(f"[ERROR] DynamicsVessel3D.derivatives failed: {e}")
            raise