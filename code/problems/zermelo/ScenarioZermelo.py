# ===============================================================================
# ScenarioZermelo Class
#
# Author: José Antonio González Prieto
# Date: 10/08/2025
# Version: 1.0
# Description:
#   Strongly typed implementation of the state dynamics for Zermelo's navigation
#   problem within a 2D domain, under constant or spatially varying current fields.
#   The vessel evolves according to given control inputs (forward speed and
#   angular rate) while being influenced by environmental currents provided by a
#   current-field model. This class extends a generic Scenario base class by
#   incorporating utilities specific to goal-reaching tasks and current handling.
# ===============================================================================

from __future__ import annotations

from utils.Scenario import Scenario


class ScenarioZermelo(Scenario):
    """
    Concrete scenario for Zermelo's navigation problem. It augments the base
    :class:`Scenario` with vessel properties and a current-field model, enabling
    goal-conditioned navigation affected by ambient currents.

    Attributes
    ----------
    start : tuple[float, float]
        Initial vessel position in world coordinates (x, y).
    initial_heading : float
        Initial vessel heading in radians.
    goal : tuple[float, float]
        Target position to be reached (x, y).
    goal_radius : float
        Acceptable radius around the goal within which the target is considered reached (meters).
    current_field : object
        An object that models the current field. It is expected (by convention) to
        implement a method
        ``getCurrentAtPosition(x: float, y: float, current_velocity: float) -> tuple[float, float]``
        returning the current vector components ``(u, v)`` at the given position.
    ship_velocity : float
        Vessel speed in still water (m/s).
    r_max : float
        Maximum allowable angular rate (rad/s).
    r_min : float
        Minimum allowable angular rate (rad/s).
    max_steps : int
        Upper bound on the number of simulation steps.
    goal_distance : float
        Precomputed straight-line distance from ``start`` to ``goal``.
    """

    def __init__(
        self,
        size: tuple[int, int],
        current_field: object,
        start: tuple[float, float],
        initial_heading: float,
        goal: tuple[float, float],
        goal_radius: float,
        ship_velocity: float,
        r_max: float,
        r_min: float,
        max_steps: int,
    ) -> None:
        """
        Initialize the Zermelo scenario with geometry, vessel, and current properties.

        Parameters
        ----------
        size : tuple[int, int]
            Domain dimensions (width, height) in grid units or meters, depending on the base scenario.
        current_field : object
            Current-field provider. By convention, it should implement
            ``getCurrentAtPosition(x: float, y: float, current_velocity: float) -> tuple[float, float]``.
        start : tuple[float, float]
            Initial vessel position (x, y).
        initial_heading : float
            Initial vessel heading in radians.
        goal : tuple[float, float]
            Target position to be reached (x, y).
        goal_radius : float
            Acceptable radius around the goal (meters).
        ship_velocity : float
            Vessel speed in still water (m/s).
        r_max : float
            Maximum angular rate (rad/s).
        r_min : float
            Minimum angular rate (rad/s).
        max_steps : int
            Maximum number of simulation steps allowed.

        Returns
        -------
        None
            This constructor does not return a value.

        Raises
        ------
        Exception
            Propagates any error encountered during attribute initialization.
        """
        try:
            super().__init__(size)
            self.start: tuple[float, float] = start
            self.initial_heading: float = initial_heading
            self.goal: tuple[float, float] = goal
            self.goal_radius: float = goal_radius
            self.current_field: object = current_field
            self.ship_velocity: float = ship_velocity
            self.r_max: float = r_max
            self.r_min: float = r_min
            self.max_steps: int = max_steps
            self.goal_distance: float = self.distance(start, goal)
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.__init__ failed: {e}")
            raise

    def isGoalReached(self, x: float, y: float) -> bool:
        """
        Determine whether a position lies within the goal radius.

        Parameters
        ----------
        x : float
            X-coordinate of the position to evaluate.
        y : float
            Y-coordinate of the position to evaluate.

        Returns
        -------
        bool
            ``True`` if the Euclidean distance to ``goal`` is less than or equal to
            ``goal_radius``; ``False`` otherwise.

        Raises
        ------
        Exception
            Propagates any error encountered during distance computation.
        """
        try:
            return self.distance((x, y), self.goal) <= self.goal_radius
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.isGoalReached failed: {e}")
            raise

    def getGoalDistance(self) -> float:
        """
        Get the straight-line distance from the starting point to the goal.

        Returns
        -------
        float
            Distance from ``start`` to ``goal`` in the scenario's units.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the cached value.
        """
        try:
            return self.goal_distance
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getGoalDistance failed: {e}")
            raise

    def getMaxSteps(self) -> int:
        """
        Get the maximum number of simulation steps permitted.

        Returns
        -------
        int
            Upper bound on the number of simulation steps.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.max_steps
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getMaxSteps failed: {e}")
            raise

    def getInitialState(self) -> tuple[float, float, float]:
        """
        Get the initial vessel state.

        Returns
        -------
        tuple[float, float, float]
            Tuple ``(x, y, heading)`` representing the initial position and heading.

        Raises
        ------
        Exception
            Propagates any error encountered while constructing the state.
        """
        try:
            return (self.start[0], self.start[1], self.initial_heading)
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getInitialState failed: {e}")
            raise

    def getInitialHeading(self) -> float:
        """
        Get the initial vessel heading.

        Returns
        -------
        float
            Initial heading in radians.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.initial_heading
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getInitialHeading failed: {e}")
            raise

    def getStart(self) -> tuple[float, float]:
        """
        Get the initial coordinates of the scenario.

        Returns
        -------
        tuple[float, float]
            Starting position as ``(x, y)``.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.start
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getStart failed: {e}")
            raise

    def getShipVelocity(self) -> float:
        """
        Get the vessel's constant speed in still water.

        Returns
        -------
        float
            Ship velocity (m/s).

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.ship_velocity
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getShipVelocity failed: {e}")
            raise

    def getRMax(self) -> float:
        """
        Get the maximum turning rate.

        Returns
        -------
        float
            Maximum angular rate (rad/s).

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.r_max
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getRMax failed: {e}")
            raise

    def getRMin(self) -> float:
        """
        Get the minimum turning rate.

        Returns
        -------
        float
            Minimum angular rate (rad/s).

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.r_min
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getRMin failed: {e}")
            raise

    def getGoalRadius(self) -> float:
        """
        Get the goal radius.

        Returns
        -------
        float
            Acceptable goal radius (meters).

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.goal_radius
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getGoalRadius failed: {e}")
            raise

    def getGoal(self) -> tuple[float, float]:
        """
        Get the goal coordinates.

        Returns
        -------
        tuple[float, float]
            Goal position as ``(x, y)``.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the value.
        """
        try:
            return self.goal
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getGoal failed: {e}")
            raise

    def setCurrentField(self, current_field: object) -> None:
        """
        Set the current-field provider.

        Parameters
        ----------
        current_field : object
            Current-field model. By convention, it should implement
            ``getCurrentAtPosition(x: float, y: float, current_velocity: float) -> tuple[float, float]``.

        Returns
        -------
        None
            This method does not return a value.

        Raises
        ------
        Exception
            Propagates any error encountered while assigning the provider.
        """
        try:
            self.current_field = current_field
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.setCurrentField failed: {e}")
            raise

    def getCurrentField(self) -> object:
        """
        Get the currently assigned current-field provider.

        Returns
        -------
        object
            The current-field model associated with this scenario.

        Raises
        ------
        Exception
            Propagates any error encountered while accessing the provider.
        """
        try:
            return self.current_field
        except Exception as e:
            print(f"[ERROR] ScenarioZermelo.getCurrentField failed: {e}")
            raise
