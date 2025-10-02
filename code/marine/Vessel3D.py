# ===============================================================================
# Vessel3D Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Thin wrapper around :class:`DynamicsVessel3D` that initializes a 3-DoF vessel
#   state and default kinematic parameters. This class does not alter the base
#   dynamics; it only sets up commonly used defaults such as maximum velocity,
#   maximum heading rate (in radians), and canonical initial/goal states.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Dict, Iterable, Tuple
from dynamics.DynamicsVessel3D import DynamicsVessel3D


class Vessel3D(DynamicsVessel3D):
    """
    Convenience specialization of :class:`DynamicsVessel3D` that stores the current
    state and ensures a minimal set of default parameters for typical simulations.

    Attributes
    ----------
    state : tuple[float, float, float] | list[float]
        Current vessel state ``(x, y, heading)`` in world coordinates (radians for heading).
    parameters : dict[str, Any]
        Parameter dictionary passed to the base class, augmented with defaults:
          - ``max_velocity`` (float): default 10.0 [m/s]
          - ``max_heading_rate`` (float): default 10 deg/s converted to radians
          - ``initial_state`` (tuple): defaults to the provided ``state``
          - ``goal_state`` (tuple): defaults to ``(0.0, 0.0, 0.0)``
    """

    def __init__(self, state: Tuple[float, float, float] | Iterable[float], parameters: Dict[str, Any]) -> None:
        """
        Initialize the vessel wrapper and ensure required default parameters.

        Parameters
        ----------
        state : tuple[float, float, float] | Iterable[float]
            Initial vessel state as ``(x, y, heading)``; heading in radians.
        parameters : dict[str, Any]
            Configuration parameters for :class:`DynamicsVessel3D`. This method
            augments the dictionary with sensible defaults if missing.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error encountered during initialization.
        """
        try:
            super().__init__(parameters)
            # Preserve provided state as-is (no validation/coercion to keep logic intact)
            self.state = state  # type: ignore[assignment]
            # Defaults (only set when missing)
            self.parameters.setdefault("max_velocity", 10.0)
            self.parameters.setdefault("max_heading_rate", self.deg2rad(10.0))
            self.parameters.setdefault("initial_state", state)
            self.parameters.setdefault("goal_state", (0.0, 0.0, 0.0))
        except Exception as e:
            print(f"[ERROR] Vessel3D.__init__ failed: {e}")
            raise
