# ===============================================================================
# Problem Base Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Generic problem container that manages scenario/solver configurations,
#   orchestrates creation and solving workflows, and provides database I/O
#   helpers (schema-agnostic stubs) for scenarios, currents, simulations,
#   and solver results. This base class is intended to be subclassed by
#   concrete problems (e.g., Zermelo) that implement specific creation and
#   solution logic while reusing the common configuration and persistence API.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Optional, Iterable
import numpy as np


class Problem:
    """
    Abstract problem definition that stores shared configuration/state and exposes
    a minimal API for scenario creation, solving, and persistence.

    Attributes
    ----------
    type : str
        Problem type label (e.g., 'random', 'deterministic').
    rng_generator : np.random.Generator
        Random number generator used by the problem.
    solvers_configuration : dict[str, Any]
        Registry of solvers enabled/disabled and related flags.
    solvers_parameters : dict[str, Any]
        Global parameters injected to solver instances.
    scenario_parameters : dict[str, Any]
        Parameters used to construct/configure the scenario.
    scenario : Any | None
        The scenario instance managed by this problem.
    solvers : dict[str, Any]
        Instantiated solvers keyed by name/identifier.
    solutions_data : dict[str, Any]
        Output bundle mapping solver runs to their results.
    """

    def __init__(self, type: str, rng: np.random.Generator) -> None:
        """
        Initialize the base problem container.

        Parameters
        ----------
        type : str
            Problem type label.
        rng : np.random.Generator
            Random number generator provided by the caller.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization.
        """
        try:
            self.type: str = type
            self.rng_generator: np.random.Generator = rng
            self.solvers_configuration: dict[str, Any] = {}
            self.solvers_parameters: dict[str, Any] = {}
            self.scenario_parameters: dict[str, Any] = {}
            self.scenario: Any | None = None
            self.solvers: dict[str, Any] = {}
            self.solutions_data: dict[str, Any] = {}
            return
        except Exception as e:
            print(f"[ERROR] Problem.__init__ failed: {e}")
            raise

    def getScenario(self) -> Any | None:
        """
        Return the currently configured scenario instance.

        Returns
        -------
        object | None
            Scenario instance if available; otherwise ``None``.
        """
        try:
            return self.scenario
        except Exception as e:
            print(f"[ERROR] Problem.getScenario failed: {e}")
            raise

    def setScenarioParameter(self, parameter: str, value: Any) -> None:
        """
        Set (or override) a scenario-level parameter.

        Parameters
        ----------
        parameter : str
            Parameter name.
        value : Any
            Parameter value.

        Returns
        -------
        None
        """
        try:
            self.scenario_parameters[parameter] = value
            return
        except Exception as e:
            print(f"[ERROR] Problem.setScenarioParameter failed: {e}")
            raise

    def getScenarioParameter(self, parameter: str, default_value: Any = None) -> Any:
        """
        Retrieve a scenario-level parameter with an optional default.

        Parameters
        ----------
        parameter : str
            Parameter name.
        default_value : Any, optional
            Value returned if the parameter is not present.

        Returns
        -------
        Any
            Parameter value or the provided default.
        """
        try:
            parameter_value: Any = self.scenario_parameters.get(parameter, default_value)
            if parameter_value is None:
                parameter_value = default_value
            return parameter_value
        except Exception as e:
            print(f"[ERROR] Problem.getScenarioParameter failed: {e}")
            raise

    def setSolverParameter(self, parameter: str, value: Any) -> None:
        """
        Set (or override) a global solver parameter.

        Parameters
        ----------
        parameter : str
            Parameter name.
        value : Any
            Parameter value.

        Returns
        -------
        None
        """
        try:
            self.solvers_parameters[parameter] = value
            return
        except Exception as e:
            print(f"[ERROR] Problem.setSolverParameter failed: {e}")
            raise

    def getSolverParameter(self, parameter: str, default_value: Any = None) -> Any:
        """
        Retrieve a global solver parameter with an optional default.

        Parameters
        ----------
        parameter : str
            Parameter name.
        default_value : Any, optional
            Value returned if the parameter is not present.

        Returns
        -------
        Any
            Parameter value or the provided default.
        """
        try:
            parameter_value: Any = self.solvers_parameters.get(parameter, default_value)
            if parameter_value is None:
                parameter_value = default_value
            return parameter_value
        except Exception as e:
            print(f"[ERROR] Problem.getSolverParameter failed: {e}")
            raise

    def configure(
        self,
        scenario_parameters: dict[str, Any],
        solvers_parameters: dict[str, Any],
        solvers_configuration: dict[str, Any],
    ) -> bool:
        """
        Apply bulk configuration for scenario and solver parameters.

        Parameters
        ----------
        scenario_parameters : dict[str, Any]
            Scenario parameter dictionary.
        solvers_parameters : dict[str, Any]
            Global solver parameters dictionary.
        solvers_configuration : dict[str, Any]
            Solver configuration flags/metadata.

        Returns
        -------
        bool
            ``True`` on success.

        Raises
        ------
        Exception
            Propagates any error during configuration.
        """
        try:
            self.scenario_parameters = scenario_parameters
            self.solvers_parameters = solvers_parameters
            self.solvers_configuration = solvers_configuration
            return True
        except Exception as e:
            print(f"[ERROR] Problem.configure failed: {e}")
            raise

    def create(self, type: str = "random") -> bool:
        """
        Create or initialize a scenario according to the given type.

        Parameters
        ----------
        type : str, optional
            Creation mode (implementation-defined), default 'random'.

        Returns
        -------
        bool
            ``True`` if a scenario was created; otherwise ``False``.
        """
        try:
            return False
        except Exception as e:
            print(f"[ERROR] Problem.create failed: {e}")
            raise

    def solve(self) -> dict[str, Any]:
        """
        Execute the solving pipeline and return solution artifacts.

        Returns
        -------
        dict[str, Any]
            Structure with keys:
              - 'simulation_data': Any
              - 'solver_instance': Any
        """
        try:
            self.solutions_data = {"simulation_data": None, "solver_instance": None}
            return self.solutions_data
        except Exception as e:
            print(f"[ERROR] Problem.solve failed: {e}")
            raise

    def clearDb(self, connection: Any, cursor: Any) -> None:
        """
        Delete all rows from key tables and reset sqlite_sequence to keep
        AUTOINCREMENT counters consistent.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.

        Returns
        -------
        None

        Raises
        ------
        Exception
            On database operation failure.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Problem.clearDb failed: {e}")
            connection.rollback()
            raise

    # -------------------- CURRENTS / CURRENTS_DATA --------------------------

    def currentToDb(self, connection: Any, cursor: Any, scenario_id: int, seed: float) -> int:
        """
        Insert a row into CURRENTS and return its ID.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        scenario_id : int
            Foreign key referencing SCENARIOS.
        seed : float
            Seed associated with the current field generation.

        Returns
        -------
        int
            Newly created current_id.

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            current_id: int = 0  # cursor.lastrowid
            return current_id
        except Exception as e:
            print(f"[ERROR] Problem.currentToDb failed: {e}")
            connection.rollback()
            raise

    def currentsDataToDb(
        self,
        connection: Any,
        cursor: Any,
        current_id: int,
        time_list: list[float],
        points: list[tuple[float, float]],
        vectors: list[tuple[float, float]],
    ) -> None:
        """
        Bulk-insert current-field samples into CURRENTS_DATA.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        current_id : int
            Parent current identifier.
        time_list : list[float]
            Time stamps for each grid sample.
        points : list[tuple[float, float]]
            Grid points aligned with `time_list`.
        vectors : list[tuple[float, float]]
            Current vectors aligned with `time_list`.

        Returns
        -------
        None

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Problem.currentsDataToDb failed: {e}")
            connection.rollback()
            raise

    def loadCurrentsFromDb(self, cursor: Any, current_id: int) -> dict | None:
        """
        Load a full current field (CURRENTS + CURRENTS_DATA) by identifier.

        Parameters
        ----------
        cursor : Any
            Database cursor.
        current_id : int
            Identifier of the current field to retrieve.

        Returns
        -------
        dict | None
            Dictionary with keys:
              - 'scenario_id'
              - 'seed'
              - 'time_history'
              - 'grid_points'
              - 'current_vectors'
            or ``None`` if not found.

        Raises
        ------
        Exception
            On database query/packing failure.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Problem.loadCurrentsFromDb failed: {e}")
            raise

    # ------------------------- SCENARIOS -----------------------------------

    def scenarioToDb(
        self,
        connection: Any,
        cursor: Any,
        size_id: int,
        current_field_id: int,
        initial_heading: float,
        ship_velocity: float,
        current_velocity: float,
        goal: tuple[float, float],
        goal_radius: float,
        r_max: float,
        seed: int,
    ) -> int:
        """
        Insert a scenario row into SCENARIOS and return its ID.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        size_id : int
            Scenario size identifier.
        current_field_id : int
            Current-field type/index associated with the scenario.
        initial_heading : float
            Initial vessel heading (radians).
        ship_velocity : float
            Vessel speed in still water (m/s).
        current_velocity : float
            Nominal current magnitude (m/s).
        goal : tuple[float, float]
            Goal position (x, y).
        goal_radius : float
            Goal acceptance radius.
        r_max : float
            Maximum turning rate (rad/s).
        seed : int
            Seed for scenario generation.

        Returns
        -------
        int
            Newly created scenario_id.

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            scenario_id: int = 0  # cursor.lastrowid
            return scenario_id
        except Exception as e:
            print(f"[ERROR] Problem.scenarioToDb failed: {e}")
            connection.rollback()
            raise

    def loadScenarioFromDb(self, cursor: Any, scenario_id: int) -> dict | None:
        """
        Load scenario parameters from SCENARIOS by identifier.

        Parameters
        ----------
        cursor : Any
            Database cursor.
        scenario_id : int
            Scenario identifier.

        Returns
        -------
        dict | None
            Scenario dictionary with keys:
              - size_id, current_field_id, initial_heading, ship_velocity,
                current_velocity, goal (x, y), goal_radius, r_max, seed
            or ``None`` if not found.

        Raises
        ------
        Exception
            On database query/packing failure.
        """
        try:
            row = range(10)  # placeholder stub
            return {
                "size_id": row[0],
                "current_field_id": row[1],
                "initial_heading": row[2],
                "ship_velocity": row[3],
                "current_velocity": row[4],
                "goal": (row[5], row[6]),
                "goal_radius": row[7],
                "r_max": row[8],
                "seed": row[9],
            }
        except Exception as e:
            print(f"[ERROR] Problem.loadScenarioFromDb failed: {e}")
            raise

    # ----------------------- SIMULATIONS / SIMULATIONS_DATA -----------------

    def simulationToDb(
        self,
        connection: Any,
        cursor: Any,
        scenario_id: int,
        solver_id: int,
        total_time: float,
        total_distance: float,
        num_steps: int,
        execution_time: float,
        goal_objective: int,
        distance_to_goal: float,
        navegation_index: float,
        solver_seed: int,
    ) -> int:
        """
        Insert a summary row into SIMULATIONS (including solver_seed) and return its ID.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        scenario_id : int
            Parent scenario identifier.
        solver_id : int
            Solver identifier.
        total_time : float
            Total simulated time.
        total_distance : float
            Traveled distance.
        num_steps : int
            Number of integration steps.
        execution_time : float
            Wall-clock time spent by the solver.
        goal_objective : int
            1 if the goal was reached; 0 otherwise.
        distance_to_goal : float
            Terminal distance to goal.
        navegation_index : float
            Composite performance index.
        solver_seed : int
            Seed used by the solver run.

        Returns
        -------
        int
            Newly created simulation_id.

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            simulation_id: int = 0  # cursor.lastrowid
            return simulation_id
        except Exception as e:
            print(f"[ERROR] Problem.simulationToDb failed: {e}")
            connection.rollback()
            raise

    def simulationsDataToDb(
        self,
        connection: Any,
        cursor: Any,
        simulation_id: int,
        simulation_data: dict,
    ) -> None:
        """
        Bulk-insert time histories into SIMULATIONS_DATA using executemany.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        simulation_id : int
            Parent simulation identifier.
        simulation_data : dict
            Full time-series payload as produced by solvers/dynamics.

        Returns
        -------
        None

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Problem.simulationsDataToDb failed: {e}")
            connection.rollback()
            raise

    def solverDataToDb(
        self,
        connection: Any,
        cursor: Any,
        solver_id: int,
        scenario_id: int,
        simulation_data: dict | None,
        solver_: Any,
    ) -> int:
        """
        Insert solver metadata and time-series for a single execution.

        Parameters
        ----------
        connection : Any
            Open database connection.
        cursor : Any
            Database cursor.
        solver_id : int
            Solver identifier.
        scenario_id : int
            Parent scenario identifier.
        simulation_data : dict | None
            Full time-series payload; may be None if the run failed.
        solver_ : Any
            Solver instance reference.

        Returns
        -------
        int
            Newly created simulation_id associated to this solver run.

        Raises
        ------
        Exception
            On database insertion failure.
        """
        try:
            simulation_id: int = 0
            return simulation_id
        except Exception as e:
            print(f"[ERROR] Problem.solverDataToDb failed: {e}")
            raise

    def loadSolverResultsFromDb(self, cursor: Any, scenario_id: int) -> list[dict]:
        """
        Load all simulations bound to a given scenario_id, including time-series.

        Parameters
        ----------
        cursor : Any
            Database cursor.
        scenario_id : int
            Scenario identifier.

        Returns
        -------
        list[dict]
            List of dictionaries each containing:
              - 'solver_name'
              - 'simulation_data'
              - 'solver_instance'

        Raises
        ------
        Exception
            On database query/packing failure.
        """
        try:
            results: list[dict] = {}
            return results
        except Exception as e:
            print(f"[ERROR] Problem.loadSolverResultsFromDb failed: {e}")
            raise
