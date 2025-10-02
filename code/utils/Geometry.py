# ===============================================================================
# Geometry Utility Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This module provides a `Geometry` class with a collection of common
#   geometric calculation utilities. It includes functions for distance,
#   angles, coordinate transformations, and trajectory analysis. Several
#   core calculations are JIT-compiled with Numba for high performance.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import List, Tuple
import numpy as np
from numba import njit

# =====================================================
# Numba-accelerated Helper Functions
# =====================================================

@njit
def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    x1: float = p1[0]
    y1: float = p1[1]
    x2: float = p2[0]
    y2: float = p2[1]
    return np.hypot(x2 - x1, y2 - y1)

@njit
def _angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate the angle in radians of the vector from p1 to p2."""
    x1: float = p1[0]
    y1: float = p1[1]
    x2: float = p2[0]
    y2: float = p2[1]
    return np.atan2(y2 - y1, x2 - x1)

@njit
def _localToGlobal(origin: Tuple[float, float, float], local: Tuple[float, float]) -> Tuple[float, float]:
    """Transform a point from a local to a global reference frame."""
    x0: float; y0: float; theta0: float
    x0, y0, theta0 = origin
    xl: float; yl: float
    xl, yl = local
    cos_t: float = np.cos(theta0)
    sin_t: float = np.sin(theta0)
    xg: float = x0 + xl * cos_t - yl * sin_t
    yg: float = y0 + xl * sin_t + yl * cos_t
    return (xg, yg)

@njit
def _globalToLocal(origin: Tuple[float, float, float], global_p: Tuple[float, float]) -> Tuple[float, float]:
    """Transform a point from a global to a local reference frame."""
    x0: float; y0: float; theta0: float
    x0, y0, theta0 = origin
    xg: float; yg: float
    xg, yg = global_p
    xt: float = xg - x0
    yt: float = yg - y0
    cos_t: float = np.cos(theta0)
    sin_t: float = np.sin(theta0)
    xl: float = xt * cos_t + yt * sin_t
    yl: float = -xt * sin_t + yt * cos_t
    return (xl, yl)

@njit
def _angleWrap(angle: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# =====================================================
# Main Geometry Class
# =====================================================

class Geometry():
    """Provides a set of methods for geometric calculations."""

    def __init__(self):
        """Initialize the Geometry instance."""
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Geometry.__init__ failed: {e}")
            raise

    def calculateTrajectoryDistance(self, states_history: List[Tuple[float, float, float]]) -> float:
        """Calculate the total distance of a trajectory.

        Parameters
        ----------
        states_history : List[Tuple[float, float, float]]
            A list of state tuples (x, y, theta) representing the trajectory.

        Returns
        -------
        float
            The total Euclidean distance traveled along the path.
        """
        try:
            states_array: np.ndarray = np.array(states_history)
            deltas: np.ndarray = np.diff(states_array[:, :2], axis=0)
            total_distance: float = float(np.sum(np.linalg.norm(deltas, axis=1)))
            return total_distance
        except Exception as e:
            print(f"[ERROR] Geometry.calculateTrajectoryDistance failed: {e}")
            raise

    def generateCircleNonOverlappingPoints(
        self,
        R: float,
        r: float,
        n: int,
        rng: np.random.Generator,
        mode: str = "random",
        max_attempts: int = 10000,
        epsilon: float = 1e-6
    ) -> List[Tuple[float, float]]:
        """Generate n non-overlapping points on a circle.

        Places n points on a circle of radius R such that disks of radius r
        centered at those points do not overlap.

        Parameters
        ----------
        R : float
            Radius of the main circle on which points are placed.
        r : float
            Radius of the disks around each point used for overlap checking.
        n : int
            The number of points to generate.
        rng : np.random.Generator
            The random number generator instance for reproducibility.
        mode : str, optional
            Generation mode: 'random' or 'deterministic', by default 'random'.
        max_attempts : int, optional
            Maximum attempts for the 'random' mode, by default 10000.
        epsilon : float, optional
            Small tolerance for floating point comparisons, by default 1e-6.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (x, y) tuples for the generated points.
        """
        try:
            if not (R > r > 0.0):
                raise ValueError("Requires R > r > 0.")
            if n < 1:
                raise ValueError("n must be >= 1.")

            ratio: float = min(1.0, r / R)
            delta_theta_min: float = 2.0 * np.arcsin(ratio)

            n_max: float = int(np.floor((2.0 * np.pi) / delta_theta_min)) if delta_theta_min > 0 else np.inf
            if np.isfinite(n_max) and n > n_max:
                raise ValueError(f"Cannot place {n} points without overlap: max is {n_max}.")

            def angularDistance(t: float, thetas: np.ndarray) -> float:
                if thetas.size == 0:
                    return np.inf
                d: np.ndarray = np.abs(t - thetas) % (2.0 * np.pi)
                d = np.minimum(d, 2.0 * np.pi - d)
                return float(d.min())

            thetas: np.ndarray
            if mode == "deterministic":
                theta0: float = rng.uniform(0.0, 2.0 * np.pi)
                step: float = 2.0 * np.pi / n
                thetas = (theta0 + step * np.arange(n)) % (2.0 * np.pi)
            elif mode == "random":
                thetas_list: List[float] = []
                attempts: int = 0
                while len(thetas_list) < n and attempts < max_attempts:
                    attempts += 1
                    t: float = rng.uniform(0.0, 2.0 * np.pi)
                    if angularDistance(t, np.array(thetas_list)) >= delta_theta_min - epsilon:
                        thetas_list.append(t)
                if len(thetas_list) < n:
                    raise RuntimeError("Failed to generate points within max_attempts.")
                thetas = np.sort(np.array(thetas_list))
            else:
                raise ValueError("mode must be 'random' or 'deterministic'.")

            xs: np.ndarray = R * np.cos(thetas)
            ys: np.ndarray = R * np.sin(thetas)
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
        except Exception as e:
            print(f"[ERROR] Geometry.generateCircleNonOverlappingPoints failed: {e}")
            raise
    
    def radToDeg(self, angle: float) -> float:
        """Convert an angle from radians to degrees."""
        try:
            return angle * 180.0 / np.pi
        except Exception as e:
            print(f"[ERROR] Geometry.radToDeg failed: {e}")
            raise
    
    def degToRad(self, angle: float) -> float:
        """Convert an angle from degrees to radians."""
        try:
            return angle * np.pi / 180.0
        except Exception as e:
            print(f"[ERROR] Geometry.degToRad failed: {e}")
            raise
    
    def norm(self, vector: Tuple[float, float]) -> float:
        """Calculate the L2 norm (Euclidean length) of a 2D vector."""
        try:
            return self.distance((0.0, 0.0), vector)
        except Exception as e:
            print(f"[ERROR] Geometry.norm failed: {e}")
            raise
    
    def angleWrap(self, angle: float) -> float:
        """Wrap an angle in radians to the interval [-pi, pi]."""
        try:
            return _angleWrap(angle)
        except Exception as e:
            print(f"[ERROR] Geometry.angleWrap failed: {e}")
            raise

    def generateRandomPosition(self, rng: np.random.Generator, min_distance: float, max_distance: float) -> Tuple[float, float]:
        """Generate a random 2D position within a specified radial range."""
        try:
            angle: float = rng.uniform(-np.pi, np.pi)
            modulus: float = rng.uniform(min_distance, max_distance)
            return (modulus * np.cos(angle), modulus * np.sin(angle))
        except Exception as e:
            print(f"[ERROR] Geometry.generateRandomPosition failed: {e}")
            raise

    def localToGlobalCoordinates(self, origin: Tuple[float, float, float], local_coordinates: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a point from a local to a global reference frame."""
        try:
            return _localToGlobal(origin, local_coordinates)
        except Exception as e:
            print(f"[ERROR] Geometry.localToGlobalCoordinates failed: {e}")
            raise

    def globalToLocalCoordinates(self, origin: Tuple[float, float, float], global_coordinates: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a point from a global to a local reference frame."""
        try:
            return _globalToLocal(origin, global_coordinates)
        except Exception as e:
            print(f"[ERROR] Geometry.globalToLocalCoordinates failed: {e}")
            raise

    def distanceAndAngle(self, from_point: Tuple[float, float], to_point: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate both the distance and angle between two points."""
        try:
            return (_distance(from_point, to_point), _angle(from_point, to_point))
        except Exception as e:
            print(f"[ERROR] Geometry.distanceAndAngle failed: {e}")
            raise

    def distance(self, from_point: Tuple[float, float], to_point: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points."""
        try:
            return _distance(from_point, to_point)
        except Exception as e:
            print(f"[ERROR] Geometry.distance failed: {e}")
            raise

    def angle(self, from_point: Tuple[float, float], to_point: Tuple[float, float]) -> float:
        """Calculate the angle in radians of the vector between two points."""
        try:
            return _angle(from_point, to_point)
        except Exception as e:
            print(f"[ERROR] Geometry.angle failed: {e}")
            raise