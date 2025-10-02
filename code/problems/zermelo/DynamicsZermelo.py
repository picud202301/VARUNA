# ===============================================================================
# DynamicsZermelo Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Dynamics model for Zermelo's navigation problem. The model computes the time
#   derivatives of the vessel state given control inputs and environmental
#   disturbances (currents). It leverages the DynamicsVessel3D base class for
#   core kinematic and dynamic formulations and exposes a typed, minimal API.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Dict
from dynamics.DynamicsVessel3D import DynamicsVessel3D


class DynamicsZermelo(DynamicsVessel3D):
    """
    Specialized dynamics wrapper for Zermelo's navigation problem, built on top
    of :class:`DynamicsVessel3D`. It configures the base dynamics with the
    appropriate state/control/disturbance dimensions and parameterization.

    Notes
    -----
    This class does not alter the underlying dynamic equations; it delegates
    entirely to the base class while providing a focused entry point for Zermelo
    scenarios.

    Attributes
    ----------
    (Inherited)
        All attributes and methods are inherited from :class:`DynamicsVessel3D`.
    """

    def __init__(self, num_states: int, num_controls: int, num_disturbances: int, parameters: Dict[str, Any]) -> None:
        """
        Initialize the Zermelo dynamics model.

        Parameters
        ----------
        num_states : int
            Number of state variables in the system.
        num_controls : int
            Number of control inputs (e.g., surge speed and heading rate).
        num_disturbances : int
            Number of external disturbance components (e.g., current in x/y).
        parameters : dict[str, Any]
            Configuration parameters required by the base dynamics model.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error encountered during base-class initialization.
        """
        try:
            super().__init__(
                num_states=num_states,
                num_controls=num_controls,
                num_disturbances=num_disturbances,
                parameters=parameters,
            )
        except Exception as e:
            print(f"[ERROR] DynamicsZermelo.__init__ failed: {e}")
            raise
