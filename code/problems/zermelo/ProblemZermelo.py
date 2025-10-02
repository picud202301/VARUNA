# ===============================================================================
# ProblemZermelo Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This class orchestrates the setup, execution, and data management for the
#   Zermelo Navigation Problem. It is responsible for generating random scenarios
#   based on specified parameters, configuring and running various solvers,
#   and persisting the results into a database for analysis and reproducibility.
#   The class encapsulates all logic related to a specific problem instance,
#   from initial configuration to final data storage.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import time
import os
import copy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from utils.Problem import Problem
from utils.Geometry import Geometry
from problems.zermelo.ScenarioZermelo import ScenarioZermelo
from utils.RandomNumberGenerator import RandomNumberGenerator
from marine.CurrentField import CurrentField
from problems.zermelo.PlotterZermelo import PlotterZermelo


class ProblemZermelo(Problem):
    """
    Manages the entire lifecycle of a Zermelo Navigation Problem instance,
    including scenario generation, solver execution, and database interaction.
    """
    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize the ProblemZermelo instance.

        Parameters
        ----------
        rng : np.random.Generator
            A NumPy random number generator for reproducibility.

        """
        try:
            super().__init__(type='zermelo', rng=rng)
            self.rng_generators: Dict[str, RandomNumberGenerator] = self.rng_generator.createGenerators(
                identifiers=["scenario", "current_field", "solvers"]
            )
            self.current_field: Optional[CurrentField] = None
            self.solutions_data: Optional[Dict[str, Dict[str, Any]]] = None
            self.title: str = 'ZERMELO PROBLEM'
            return
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.__init__ failed: {e}")
            raise

    def configure(self, scenario_parameters: Dict[str, Any], solvers_parameters: Dict[str, Any], solvers_configuration: Dict[str, Any]) -> bool:
        """Configure the problem instance with scenario and solver parameters.

        This method sets up the environment size, time steps, and current field
        properties based on the provided configuration dictionaries.

        Parameters
        ----------
        scenario_parameters : dict[str, Any]
            A dictionary containing parameters for the scenario.
        solvers_parameters : dict[str, Any]
            A dictionary containing parameters for the solvers.
        solvers_configuration : dict[str, Any]
            A dictionary containing the configuration for each solver.

        Returns
        -------
        bool
            True if the configuration was successful.

        Raises
        ------
        Exception
            If any error occurs during configuration.

        """
        try:
            super().configure(scenario_parameters=scenario_parameters,
                              solvers_parameters=solvers_parameters,
                              solvers_configuration=solvers_configuration)

            size_id: int = self.getScenarioParameter('size_id', 1)
            self.scenario_parameters: Dict[str, Any] = {}
            self.solvers_parameters: Dict[str, Any] = {}
            self.setScenarioParameter('size_id', size_id)
            if size_id == 1:
                self.scenario_parameters['size'] = (200.0, 200.0)
                self.solvers_parameters['time_step'] = 0.5
                self.solvers_parameters['max_execution_time'] = 3.0
                self.scenario_parameters['current_grid_size'] = 5.0
            elif size_id == 2:
                self.scenario_parameters['size'] = (2000.0, 2000.0)
                self.solvers_parameters['time_step'] = 5.0
                self.solvers_parameters['max_execution_time'] = 4.0
                self.scenario_parameters['current_grid_size'] = 50.0
            else:
                self.scenario_parameters['size'] = (20000.0, 20000.0)
                self.solvers_parameters['time_step'] = 50.0
                self.solvers_parameters['max_execution_time'] = 5.0
                self.scenario_parameters['current_grid_size'] = 500.0

            scenario_size: Tuple[float, float] = self.getScenarioParameter('size', default_value=(200.0, 200.0))
            current_type: str = self.getScenarioParameter('current_type', default_value='uniform')
            currents_json_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'marine/currents.json')
            self.current_field: CurrentField = CurrentField(
                scenario_size=scenario_size,
                current_type=current_type,
                rng=self.rng_generators.get('current_field'),
                config_file=currents_json_path
            )
            return True
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.configure failed: {e}")
            raise

    def getCurrentField(self) -> Optional[CurrentField]:
        """Retrieve the current field object associated with this problem.

        Returns
        -------
        Optional[CurrentField]
            The CurrentField instance for this problem, or None if not configured.

        Raises
        ------
        Exception
            If any error occurs during retrieval.
            
        """
        try:
            return self.current_field
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.getCurrentField failed: {e}")
            raise

    def print(self) -> None:
        """Print a formatted summary of the problem's configuration.

        This includes seed values, scenario dimensions, current properties,
        ship characteristics, and simulation conditions.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during printing.

        """
        try:
            print(f"$$$$$$$$$$$$$$$$$$$$$$ {self.title} $$$$$$$$$$$$$$$$$$$$$$")
            print("___________________________SEED___________________________________")
            print("MASTER_SEED           =", self.rng_generator.seed)
            print("___________________________SCENARIO_______________________________")
            print("SIZE ID               =", self.getScenarioParameter('size_id'))
            print("SCENARIO SIZE         =", self.getScenarioParameter('size'))
            print("GOAL                  =", self.getScenario().getGoal())
            print("INITIAL HEADING (deg) =", round(self.getScenario().getInitialHeading() * 180.0 / np.pi, 2))
            print("___________________________CURRENT_________________________________")
            print("CURRENT TYPE          =", self.getCurrentField().getCurrentType())
            print("CURRENT VELOCITY      =", self.getCurrentField().getVelocity())
            print("____________________________SHIP___________________________________")
            print("SHIP VELOCITY         =", self.getScenario().getShipVelocity())
            print("R_MAX (deg)           =", round(self.getScenario().getRMax() * 180.0 / np.pi, 2))
            print("_________________________CONDITIONS________________________________")
            print("GOAL RADIUS           =", self.getScenario().getGoalRadius())
            print("MAX STEPS             =", self.getScenario().getMaxSteps())
            print("TIME STEP             =", self.getSolverParameter('time_step'))
            print("MAX TIME              =", self.getSolverParameter('max_time'))
            print("MAX EXECUTION TIME    =", self.getSolverParameter('max_execution_time'))
            print("........................................................................")
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.print failed: {e}")
            raise

    def _create(self, type: str = 'random') -> None:
        """Generate the detailed parameters for a new scenario.

        This private method creates randomized or fixed parameters for ship velocity,
        currents, goal position, and other simulation constraints.

        Parameters
        ----------
        type : str, optional
            The type of scenario to create ('random' or fixed), by default 'random'.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during parameter creation.
            
        """
        try:
            time_step: float = self.getSolverParameter('time_step')
            scenario_size: Tuple[float, float] = self.getScenarioParameter('size')
            start: Tuple[float, float] = (0.0, 0.0)
            if type == 'random':
                ship_velocity: float = float(self.rng_generator.uniform(5.0, 15.0))
                current_velocity: float = float(self.rng_generator.uniform(1.0, 4.0))
                r_max: float = float(self.rng_generator.uniform(5.0, 15.0)) * np.pi / 180.0
                r_min: float = -r_max
                goal_radius: float = 1.75 * time_step * ship_velocity
                initial_heading: float = float(self.rng_generator.uniform(-np.pi, np.pi))
            else:
                ship_velocity: float = 10.0
                current_velocity: float = 3.0
                r_max: float = 10.0 * np.pi / 180.0
                r_min: float = -r_max
                goal_radius: float = 1.75 * time_step * ship_velocity
                initial_heading: float = float(self.rng_generator.uniform(-np.pi, np.pi))

            min_distance: float = 0.75 * max(scenario_size)
            max_distance: float = 0.95 * max(scenario_size)
            phi: float = float(self.rng_generator.uniform(-np.pi, np.pi))
            radius: float = float(self.rng_generator.uniform(min_distance, max_distance))
            goal: Tuple[float, float] = (
                round(start[0] + radius * np.cos(phi), 2),
                round(start[1] + radius * np.sin(phi), 6),
            )
            goal_distance: float = Geometry().distance(start, goal)
            max_time: int = int(5.0 * goal_distance / ship_velocity)
            max_steps: int = int(max_time / time_step)

            self.getCurrentField().setVelocity(current_velocity)
            current_type: Optional[str] = self.getScenarioParameter('current_type', default_value=None)
            if current_type is None:
                current_types: List[str] = CurrentField.getCurrentTypes()
                ct_choice_idx: int = self.rng_generator.integers(low=0, high=len(current_types), size=1)[0]
                current_type: str = current_types[ct_choice_idx]

            self.setSolverParameter('max_steps', max_steps)
            self.setSolverParameter('max_time', max_time)
            self.setScenarioParameter('goal', goal)
            self.setScenarioParameter('start', start)
            self.setScenarioParameter('ship_velocity', ship_velocity)
            self.setScenarioParameter('current_velocity', current_velocity)
            self.setScenarioParameter('r_max', r_max)
            self.setScenarioParameter('r_min', r_min)
            self.setScenarioParameter('goal_radius', goal_radius)
            self.setScenarioParameter('initial_heading', initial_heading)
            self.setScenarioParameter('current_type', current_type)
            self.setScenarioParameter('goal_distance', goal_distance)

            self.getCurrentField().setVelocity(current_velocity)
            self.getCurrentField().setCurrentType(current_type)

        except Exception as e:
            print(f"[ERROR] ProblemZermelo._create failed: {e}")
            raise

    def create(self, type: str = 'random') -> bool:
        """Create a complete scenario instance based on the generated parameters.

        Parameters
        ----------
        type : str, optional
            The type of scenario to create ('random' or fixed), by default 'random'.

        Returns
        -------
        bool
            True if the scenario was created successfully.

        Raises
        ------
        Exception
            If any error occurs during scenario creation.
            
        """
        try:
            self._create(type=type)

            self.scenario: ScenarioZermelo = ScenarioZermelo(
                size=self.getScenarioParameter('size'),
                current_field=copy.deepcopy(self.current_field),
                start=self.getScenarioParameter('start'),
                initial_heading=self.getScenarioParameter('initial_heading'),
                goal=self.getScenarioParameter('goal'),
                goal_radius=self.getScenarioParameter('goal_radius'),
                ship_velocity=self.getScenarioParameter('ship_velocity'),
                r_max=self.getScenarioParameter('r_max'),
                r_min=self.getScenarioParameter('r_min'),
                max_steps=self.getSolverParameter('max_steps')
            )
            return True
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.create failed: {e}")
            raise

    def summary(self, plotter: PlotterZermelo) -> None:
        """Print a summary of solutions and add them to the plotter.

        Parameters
        ----------
        plotter : PlotterZermelo
            The plotter instance to which successful solutions will be added.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during the summary creation.
            
        """
        try:
            print("%%%%%%%%%%%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%%%%%%")
            if self.solutions_data is not None:
                for solver_name, solution_solver_data in self.solutions_data.items():
                    simulation_data: Optional[Dict[str, Any]] = solution_solver_data['simulation_data']
                    solver_instance: Any = solution_solver_data['solver_instance']
                    if simulation_data is None:
                        continue
                    if simulation_data.get("goal_objective", False):
                        plotter.addSolution(simulation_data, solver_instance)
                    solver_instance.summarySimulationData(simulation_data)
            else:
                print("SOLUTION NOT FOUND!!")
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.summary failed: {e}")
            raise

    def plot(self) -> None:
        """Generate and display a plot of the scenario and its solutions.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during plotting.
            
        """
        try:
            plotter: PlotterZermelo = PlotterZermelo(
                parameters={
                    "current_quiver_scale": 50,
                    "current_quiver_width": 0.006,
                    "current_quiver_num_vectors": 20,
                }
            )
            self.summary(plotter=plotter)
            plotter.plot(self.getScenario())
            plotter.show()
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.plot failed: {e}")
            raise

    def solve(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Execute all active solvers on the configured scenario.

        Iterates through the configured solvers, runs them, and stores
        the resulting simulation data.

        Returns
        -------
        Optional[Dict[str, Dict[str, Any]]]
            A dictionary containing the simulation data and solver instances
            for each executed solver.

        Raises
        ------
        Exception
            If any error occurs during the solving process.
            
        """
        try:
            self.solutions_data: Dict[str, Dict[str, Any]] = {}
            for solver_name, solver_configuration in self.solvers_configuration.items():
                if solver_configuration.get('active', False):
                    solver_class: Optional[Any] = solver_configuration.get('class', None)
                    if solver_class is not None:
                        parameters: Dict[str, Any] = solver_configuration.get('parameters', {})
                        scenario: ScenarioZermelo = copy.deepcopy(self.scenario)
                        solver_color: str = parameters.get('color', 'grey')
                        if parameters.get('library', 'np') == 'pyo':
                            scenario.getCurrentField().setLibraryType(library_type='pyo')
                        solver_instance: Any = solver_class(scenario=scenario, rng=self.rng_generators.get('solvers'), parameters=parameters)
                        start_time: float = time.perf_counter()
                        simulation_data: Optional[Dict[str, Any]] = solver_instance.solve(
                            max_steps=self.getSolverParameter('max_steps'),
                            time_step=self.getSolverParameter('time_step'),
                            max_execution_time=self.getSolverParameter('max_execution_time')
                        )
                        if simulation_data is not None:
                            simulation_data['execution_time'] = time.perf_counter() - start_time
                            simulation_data['color'] = solver_color
                            simulation_data = simulation_data | parameters
                            self.solutions_data[solver_name] = {'simulation_data': simulation_data, 'solver_instance': solver_instance}
            return self.solutions_data
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.solve failed: {e}")
            raise

    # =======================================================================
    # Database functions
    # =======================================================================

    def clearDb(self, connection: Any, cursor: Any) -> None:
        """Delete all rows from key tables and reset autoincrementing IDs.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during the database clearing operation.
            
        """
        try:
            cursor.execute("DELETE FROM SIMULATIONS_DATA;")
            cursor.execute("DELETE FROM SIMULATIONS;")
            cursor.execute("DELETE FROM SCENARIOS;")
            cursor.execute("DELETE FROM CURRENTS_DATA;")
            cursor.execute("DELETE FROM CURRENTS;")

            cursor.execute("DELETE FROM sqlite_sequence WHERE name='SIMULATIONS_DATA';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='SIMULATIONS';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='SCENARIOS';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='CURRENTS_DATA';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='CURRENTS';")

            connection.commit()
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.clearDb failed: {e}")
            connection.rollback()
            raise

    def currentToDb(self, connection: Any, cursor: Any, scenario_id: int, seed: int) -> int:
        """Insert a record into the CURRENTS table.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        scenario_id : int
            The foreign key linking to the SCENARIOS table.
        seed : int
            The seed used to generate this current field.

        Returns
        -------
        int
            The ID of the newly inserted current record.

        Raises
        ------
        Exception
            If any error occurs during the insertion.
            
        """
        try:
            cursor.execute(
                """
                INSERT INTO CURRENTS (scenario_id, seed)
                VALUES (?, ?);
                """,
                (scenario_id, seed),
            )
            current_id: int = cursor.lastrowid
            connection.commit()
            return current_id
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.currentToDb failed: {e}")
            connection.rollback()
            raise

    def currentsDataToDb(
        self,
        connection: Any,
        cursor: Any,
        current_id: int,
        time_list: List[float],
        points: List[Tuple[float, float]],
        vectors: List[Tuple[float, float]],
    ) -> None:
        """Insert multiple rows of time-series data into CURRENTS_DATA.

        The `time_list`, `points`, and `vectors` lists must have the same
        length and aligned order.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        current_id : int
            The foreign key linking to the CURRENTS table.
        time_list : List[float]
            A list of timestamps.
        points : List[Tuple[float, float]]
            A list of (x, y) coordinate tuples.
        vectors : List[Tuple[float, float]]
            A list of (cx, cy) current vector tuples.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during the bulk insertion.
            
        """
        try:
            rows: List[Tuple[int, str, float, float, float, float]] = [
                (current_id, str(t), float(x), float(y), float(cx), float(cy))
                for t, (x, y), (cx, cy) in zip(time_list, points, vectors)
            ]
            cursor.executemany(
                """
                INSERT INTO CURRENTS_DATA (
                    current_id, time, x, y, current_x, current_y
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            connection.commit()
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.currentsDataToDb failed: {e}")
            connection.rollback()
            raise

    def loadCurrentsFromDb(self, cursor: Any, current_id: int) -> Optional[Dict[str, Any]]:
        """Load a complete current field from the database.

        Parameters
        ----------
        cursor : Any
            The database cursor object.
        current_id : int
            The ID of the current field to load.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary containing the loaded data, or None if not found.

        Raises
        ------
        Exception
            If any error occurs during data loading.
            
        """
        try:
            cursor.execute(
                "SELECT scenario_id, seed FROM CURRENTS WHERE id = ?;",
                (current_id,),
            )
            row: Optional[Tuple] = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                """
                SELECT time, x, y, current_x, current_y
                FROM CURRENTS_DATA
                WHERE current_id = ?
                ORDER BY time ASC;
                """,
                (current_id,),
            )
            time_history: List[float] = []
            grid_points: List[Tuple[float, float]] = []
            current_vectors: List[Tuple[float, float]] = []
            for t, x, y, cx, cy in cursor.fetchall():
                time_history.append(float(t))
                grid_points.append((float(x), float(y)))
                current_vectors.append((float(cx), float(cy)))

            return {
                "scenario_id": row[0],
                "seed": row[1],
                "time_history": time_history,
                "grid_points": grid_points,
                "current_vectors": current_vectors,
            }
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.loadCurrentsFromDb failed: {e}")
            raise

    def scenarioToDb(self,
        connection: Any,
        cursor: Any,
        size_id: int,
        current_field_id: int,
        initial_heading: float,
        ship_velocity: float,
        current_velocity: float,
        goal: Tuple[float, float],
        goal_radius: float,
        r_max: float,
        seed: int,
    ) -> int:
        """Insert a scenario record into the SCENARIOS table.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        size_id : int
            The identifier for the scenario size.
        current_field_id : int
            The foreign key for the associated current field.
        initial_heading : float
            The initial heading of the ship in radians.
        ship_velocity : float
            The velocity of the ship.
        current_velocity : float
            The magnitude of the current velocity.
        goal : Tuple[float, float]
            A tuple (x, y) representing the goal coordinates.
        goal_radius : float
            The radius of the goal area.
        r_max : float
            The maximum turning rate of the ship.
        seed : int
            The master seed used for the scenario generation.

        Returns
        -------
        int
            The ID of the newly inserted scenario record.

        Raises
        ------
        Exception
            If any error occurs during insertion.
            
        """
        try:
            cursor.execute(
                """
                INSERT INTO SCENARIOS (
                    size_id, current_field_id, initial_heading, ship_velocity, current_velocity,
                    goal_x, goal_y, goal_radius, r_max, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    int(size_id),
                    int(current_field_id),
                    float(initial_heading),
                    float(ship_velocity),
                    float(current_velocity),
                    float(goal[0]),
                    float(goal[1]),
                    float(goal_radius),
                    float(r_max),
                    int(seed),
                ),
            )
            scenario_id: int = cursor.lastrowid
            connection.commit()
            return scenario_id
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.scenarioToDb failed: {e}")
            connection.rollback()
            raise

    def loadScenarioFromDb(self, cursor: Any, scenario_id: int) -> Optional[Dict[str, Any]]:
        """Load scenario parameters from the SCENARIOS table.

        Parameters
        ----------
        cursor : Any
            The database cursor object.
        scenario_id : int
            The ID of the scenario to load.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary containing the scenario parameters, or None if not found.

        Raises
        ------
        Exception
            If any error occurs during data loading.
            
        """
        try:
            cursor.execute(
                """
                SELECT size_id, current_field_id, initial_heading, ship_velocity, current_velocity,
                       goal_x, goal_y, goal_radius, r_max, seed
                FROM SCENARIOS
                WHERE id = ?;
                """,
                (scenario_id,),
            )
            row: Optional[Tuple] = cursor.fetchone()
            if row is None:
                return None

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
            print(f"[ERROR] ProblemZermelo.loadScenarioFromDb failed: {e}")
            raise

    def simulationToDb(self,
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
        """Insert a simulation summary record into the SIMULATIONS table.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        scenario_id : int
            Foreign key for the associated scenario.
        solver_id : int
            Identifier for the solver used.
        total_time : float
            Total time elapsed during the simulation.
        total_distance : float
            Total distance traveled.
        num_steps : int
            Number of steps in the simulation.
        execution_time : float
            Wall-clock time for the solver to run.
        goal_objective : int
            Integer (1 or 0) indicating if the goal was reached.
        distance_to_goal : float
            Final distance to the goal.
        navegation_index : float
            Performance metric for the navigation.
        solver_seed : int
            Seed used specifically for the solver's RNG.

        Returns
        -------
        int
            The ID of the newly inserted simulation record.

        Raises
        ------
        Exception
            If any error occurs during insertion.
            
        """
        try:
            cursor.execute(
                """
                INSERT INTO SIMULATIONS (
                    scenario_id, solver_id, total_time, total_distance, num_steps, execution_time,
                    goal_objective, distance_to_goal, navegation_index, solver_seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    int(scenario_id),
                    int(solver_id),
                    float(total_time),
                    float(total_distance),
                    int(num_steps),
                    float(execution_time),
                    int(goal_objective),
                    float(distance_to_goal),
                    float(navegation_index),
                    int(solver_seed),
                ),
            )
            simulation_id: int = cursor.lastrowid
            connection.commit()
            return simulation_id
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.simulationToDb failed: {e}")
            connection.rollback()
            raise

    def simulationsDataToDb(self,
        connection: Any,
        cursor: Any,
        simulation_id: int,
        simulation_data: Dict[str, Any],
    ) -> None:
        """Insert the complete time-series history into SIMULATIONS_DATA.

        This method uses `executemany` for efficient bulk insertion.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        simulation_id : int
            Foreign key for the associated simulation record.
        simulation_data : Dict[str, Any]
            A dictionary containing the simulation history lists.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during bulk insertion.
            
        """
        try:
            rows: List[Tuple] = []
            time_h: List[float] = simulation_data.get("time_history", [])
            states_h: List[Tuple] = simulation_data.get("states_history", [])
            controls_h: List[Tuple] = simulation_data.get("controls_history", [])
            disturb_h: List[Tuple] = simulation_data.get("disturbance_history", [])

            for t, state, control, disturb in zip(time_h, states_h, controls_h, disturb_h):
                x, y, heading = state
                v, r = control
                cx, cy = disturb
                rows.append((int(simulation_id), str(t), float(x), float(y), float(heading), float(cx), float(cy), float(v), float(r)))
            if rows:
                cursor.executemany(
                    """
                    INSERT INTO SIMULATIONS_DATA (
                        simulation_id, time, x, y, heading, current_x, current_y, velocity, r
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )
                connection.commit()
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.simulationsDataToDb failed: {e}")
            connection.rollback()
            raise

    def solverDataToDb(self,
        connection: Any,
        cursor: Any,
        solver_id: int,
        scenario_id: int,
        simulation_data: Optional[Dict[str, Any]],
        solver_zermelo: Any,
    ) -> int:
        """Insert metadata and time-series data for a solver execution.

        Parameters
        ----------
        connection : Any
            The database connection object.
        cursor : Any
            The database cursor object.
        solver_id : int
            The identifier for the solver.
        scenario_id : int
            The identifier for the scenario.
        simulation_data : Optional[Dict[str, Any]]
            The dictionary containing all results from the solver.
        solver_zermelo : Any
            The solver instance (currently unused).

        Returns
        -------
        int
            The ID of the created simulation record.

        Raises
        ------
        Exception
            If any error occurs during data insertion.
            
        """
        try:
            if simulation_data:
                goal_objective: int = 1 if simulation_data.get("goal_objective", False) else 0
                solver_seed: int = int(simulation_data.get("solver_seed", 0))
                simulation_id: int = self.simulationToDb(
                    connection=connection,
                    cursor=cursor,
                    scenario_id=scenario_id,
                    solver_id=solver_id,
                    total_time=simulation_data.get("total_time", 0.0),
                    total_distance=simulation_data.get("total_distance", 0.0),
                    num_steps=simulation_data.get("num_steps", 0),
                    execution_time=simulation_data.get("execution_time", 0.0),
                    goal_objective=goal_objective,
                    distance_to_goal=simulation_data.get("distance_to_goal", 0.0),
                    navegation_index=simulation_data.get("navegation_index", 0.0),
                    solver_seed=solver_seed,
                )
                self.simulationsDataToDb(
                    connection=connection,
                    cursor=cursor,
                    simulation_id=simulation_id,
                    simulation_data=simulation_data,
                )
            else:
                simulation_id: int = self.simulationToDb(
                    connection=connection,
                    cursor=cursor,
                    scenario_id=scenario_id,
                    solver_id=solver_id,
                    total_time=0.0,
                    total_distance=0.0,
                    num_steps=0,
                    execution_time=0.0,
                    goal_objective=0,
                    distance_to_goal=0.0,
                    navegation_index=0.0,
                    solver_seed=0,
                )
            return simulation_id
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.solverDataToDb failed: {e}")
            raise

    def loadSolverResultsFromDb(self, cursor: Any, scenario_id: int) -> List[Dict[str, Any]]:
        """Load all simulation results for a specific scenario ID.

        Parameters
        ----------
        cursor : Any
            The database cursor object.
        scenario_id : int
            The ID of the scenario whose results are to be loaded.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each representing a solver's result,
            including its time-series data.

        Raises
        ------
        Exception
            If any error occurs during data loading.
            
        """
        try:
            cursor.execute(
                """
                SELECT id, solver_id, total_time, total_distance, num_steps, execution_time,
                       goal_objective, distance_to_goal, navegation_index, solver_seed
                FROM SIMULATIONS
                WHERE scenario_id = ?;
                """,
                (scenario_id,),
            )
            simulations_meta: List[Tuple] = cursor.fetchall()
            results: List[Dict[str, Any]] = []

            for sim in simulations_meta:
                simulation_id: int = sim[0]
                solver_id: int = sim[1]
                solver_name: str = {0: "analytic", 1: "astar", 2: "ipopt", 3: "pso"}.get(solver_id, f"solver_{solver_id}")

                simulation_data: Dict[str, Any] = {
                    "id": simulation_id,
                    "total_time": sim[2],
                    "total_distance": sim[3],
                    "num_steps": sim[4],
                    "execution_time": sim[5],
                    "goal_objective": bool(sim[6]),
                    "distance_to_goal": sim[7],
                    "navegation_index": sim[8],
                    "solver_seed": sim[9],
                    "time_history": [],
                    "states_history": [],
                    "controls_history": [],
                    "disturbance_history": [],
                }

                cursor.execute(
                    """
                    SELECT time, x, y, heading, current_x, current_y, velocity, r
                    FROM SIMULATIONS_DATA
                    WHERE simulation_id = ?
                    ORDER BY time ASC;
                    """,
                    (simulation_id,),
                )
                for row in cursor.fetchall():
                    t: float = float(row[0])
                    x: float = float(row[1])
                    y: float = float(row[2])
                    heading: float = float(row[3])
                    cx: float = float(row[4])
                    cy: float = float(row[5])
                    v: float = float(row[6])
                    r: float = float(row[7])

                    simulation_data["time_history"].append(t)
                    simulation_data["states_history"].append((x, y, heading))
                    simulation_data["controls_history"].append((v, r))
                    simulation_data["disturbance_history"].append((cx, cy))

                results.append({
                    "solver_name": solver_name,
                    "simulation_data": simulation_data,
                    "solver_instance": None,
                })

            return results
        except Exception as e:
            print(f"[ERROR] ProblemZermelo.loadSolverResultsFromDb failed: {e}")
            raise