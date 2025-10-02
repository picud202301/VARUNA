# ===============================================================================
# Scenario Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Base class for managing scenarios. Provides size handling and helper
#   methods for retrieving dimensions. Inherits from `Geometry` to allow
#   geometric computations within scenarios.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Tuple
from utils.Geometry import Geometry


class Scenario(Geometry):
    """
    Base class for scenario management, providing size attributes and
    convenience accessors. Inherits from `Geometry` for geometric operations.
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        """Initialize a new Scenario instance.

        Parameters
        ----------
        size : tuple[int, int]
            Scenario dimensions as (width, height).
        """
        try:
            super().__init__()
            self.size_x: int = size[0]
            self.size_y: int = size[1]
        except Exception as e:
            print(f"[ERROR] Scenario.__init__ failed: {e}")
            raise

    def getMaxSize(self) -> int:
        """Get the maximum dimension of the scenario.

        Returns
        -------
        int
            The largest value between width (x) and height (y).
        """
        try:
            return max(self.size_x, self.size_y)
        except Exception as e:
            print(f"[ERROR] Scenario.getMaxSize failed: {e}")
            raise

    def getSize(self) -> Tuple[int, int]:
        """Get the dimensions of the scenario.

        Returns
        -------
        tuple[int, int]
            A tuple (width, height) representing the scenario size.
        """
        try:
            return (self.size_x, self.size_y)
        except Exception as e:
            print(f"[ERROR] Scenario.getSize failed: {e}")
            raise

    def getSizeY(self) -> int:
        """Get the vertical dimension (height) of the scenario.

        Returns
        -------
        int
            The height of the scenario.
        """
        try:
            return self.size_y
        except Exception as e:
            print(f"[ERROR] Scenario.getSizeY failed: {e}")
            raise

    def getSizeX(self) -> int:
        """Get the horizontal dimension (width) of the scenario.

        Returns
        -------
        int
            The width of the scenario.
        """
        try:
            return self.size_x
        except Exception as e:
            print(f"[ERROR] Scenario.getSizeX failed: {e}")
            raise