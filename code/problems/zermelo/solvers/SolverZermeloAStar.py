# ===============================================================================
# Zermelo Navigation Problem plotter of simulations results
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description: This script defines the `SolverZermeloAStar` class, an implementation
#              of the A* search algorithm to solve the Zermelo navigation problem.
#              It finds a time-optimal path for a ship in a dynamic current field.
#              The algorithm operates on a continuous state space, generating neighbors
#              by simulating short trajectories towards dynamically generated target points.
#              The solver iteratively refines its search strategy and can fall back on
#              an initial analytical solution if one is found quickly.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import heapq
import copy
import time
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from problems.zermelo.solvers.SolverZermeloAnalytic import SolverZermeloAnalytic
from problems.zermelo.ScenarioZermelo import ScenarioZermelo


class SolverZermeloAStar(SolverZermeloAnalytic):
    """A* solver for the Zermelo navigation problem on a continuous space.

    This class extends the analytical solver by implementing the A* search
    algorithm with a custom heuristic to find time-optimal paths.
    """

    def __init__(self, scenario: ScenarioZermelo, rng: np.random.Generator, parameters: dict) -> None:
        """Initializes the SolverZermeloAStar instance.

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
            self.id: int = 1  # Solver identification
            self.path: List[Tuple[float, ...]] = []
            self.current_to_neighbor_controls_history: Dict[Tuple, Dict[Tuple, List]] = {}
            self.controls_history: Optional[List[Tuple[float, float]]] = None
            self.num_controls_history: int = 0
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.__init__ failed: {e}")
            raise

    def heuristic(self, node: Tuple[float, ...], goal: Tuple[float, ...], add_turn_time: bool = True) -> float:
        """Calculate the heuristic cost (estimated time) from a node to the goal.

        The heuristic estimates the minimum time, considering ship speed, water
        current, and optional turning time, assuming a straight-line path.

        Parameters
        ----------
        node : Tuple[float, ...]
            The current state tuple (x, y, theta).
        goal : Tuple[float, ...]
            The goal state tuple (xg, yg, theta_g).
        add_turn_time : bool, optional
            If True, includes the minimum turning time in the estimate, by default True.

        Returns
        -------
        float
            The estimated time to reach the goal (heuristic value).

        Raises
        ------
        Exception
            Propagates any error that occurs during calculation.
        """
        try:
            x: float; y: float; theta: float
            x, y, theta = node
            xg: float; yg: float
            xg, yg, _ = goal

            r_max: float = self.scenario.getRMax()
            dx: float = xg - x
            dy: float = yg - y
            d: float = self.distance((x, y), (xg, yg))

            if d < self.scenario.getGoalRadius():
                return 0.0

            ux: float = dx / d
            uy: float = dy / d
            cx: float; cy: float
            cx, cy = self.scenario.getCurrentField().getCurrentAtPosition(x, y)
            c_dot_u: float = cx * ux + cy * uy
            V_s: float = self.scenario.getShipVelocity()
            v_parallel_best: float = V_s + c_dot_u
            max_time: float = self.max_steps * self.time_step
            base_time: float

            if v_parallel_best <= 0.0:
                base_time = max_time
            else:
                base_time = d / max(1e-1, v_parallel_best)

            if add_turn_time and r_max is not None and r_max > 0:
                beta: float = np.arctan2(uy, ux)
                delta: float = self.angleWrap((beta - theta))
                t_turn_min: float = abs(delta) / r_max
                base_time = t_turn_min + base_time

            return base_time
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.heuristic failed: {e}")
            raise

    def generateTargetPoints(self, state: Tuple[float, ...], r: Optional[float] = None) -> List[np.ndarray]:
        """Generate intermediate target points on a circle around the goal.

        This method provides a set of strategic intermediate goals for the local
        planner by covering the visible arc from the current state.

        Parameters
        ----------
        state : Tuple[float, ...]
            The current state of the ship (x, y, theta).
        r : Optional[float], optional
            The radius of the small covering circles. If None, the scenario's
            goal radius is used, by default None.

        Returns
        -------
        List[np.ndarray]
            A list of [x, y] coordinates for the generated target points.

        Raises
        ------
        Exception
            Propagates any error that occurs during generation.
        """
        try:
            goal: np.ndarray = np.array(self.scenario.getGoal())
            state_pos: np.ndarray = np.array(state[:2])

            if r is None:
                r = float(self.scenario.getGoalRadius())

            d: float = self.distance(state_pos, goal[:2])
            if d < 1e-9:
                return [goal]

            R: float = d / 2.0
            if R < 1.5 * r:
                return [goal]

            v: np.ndarray = state_pos - goal[:2]
            dv: float = self.norm(v)
            if dv < R - 1e-12:
                return [goal]

            base_angle: float = np.arctan2(v[1], v[0])
            val: float = np.clip(R / max(dv, 1e-12), -1.0, 1.0)
            alpha: float = np.arccos(val)

            a_start: float = self.angleWrap(base_angle - alpha)
            a_end: float = self.angleWrap(base_angle + alpha)

            arc_span: float = 2.0 * alpha
            if arc_span <= 1e-12:
                return [goal[:2] + R * np.array([np.cos(base_angle), np.sin(base_angle)])]

            ratio: float = np.clip(r / (2.0 * R), 0.0, 1.0)
            delta_max: float = 2.0 * np.arcsin(ratio)
            s_cover: float = 2.0 * delta_max

            if s_cover >= arc_span - 1e-12:
                center_angle: float = self.angleWrap((a_start + a_end) / 2.0 if abs(a_end - a_start) < np.pi else base_angle)
                center: np.ndarray = goal[:2] + R * np.array([np.cos(center_angle), np.sin(center_angle)])
                return [center]

            first_angle: float = self.angleWrap((base_angle - alpha) + delta_max)
            last_angle: float = self.angleWrap((base_angle + alpha) - delta_max)

            inner_span: float = (2.0 * alpha) - 2.0 * delta_max
            n_intervals: int = int(np.ceil(inner_span / s_cover))
            n_centers: int = n_intervals + 1

            def anglerange(a1: float, a2: float, n: int) -> List[float]:
                da: float = self.angleWrap(a2 - a1)
                return [self.angleWrap(a1 + da * t) for t in np.linspace(0.0, 1.0, n)]

            center_angles: List[float] = anglerange(first_angle, last_angle, n_centers)
            target_points: List[np.ndarray] = [goal[:2] + R * np.array([np.cos(a), np.sin(a)]) for a in center_angles]
            return target_points
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.generateTargetPoints failed: {e}")
            raise

    def getNeighbors(
        self, node: Tuple[float, ...], grid_time_factor: float = 1.0
    ) -> Tuple[List[Tuple[float, ...]], Dict[Tuple[float, ...], float]]:
        """Generate neighboring states for the A* search from a given node.

        It generates target points and simulates movement from the current node
        towards each target for a limited time.

        Parameters
        ----------
        node : Tuple[float, ...]
            The current state tuple (x, y, theta).
        grid_time_factor : float, optional
            A factor to scale the simulation time, by default 1.0.

        Returns
        -------
        Tuple[List[Tuple[float, ...]], Dict[Tuple[float, ...], float]]
            A tuple containing a list of neighbor states and a dictionary
            mapping each neighbor to the time taken to reach it.

        Raises
        ------
        Exception
            Propagates any error that occurs during neighbor generation.
        """
        try:
            neighbors: List[Tuple[float, ...]] = []
            neighbors_times: Dict[Tuple[float, ...], float] = {}
            target_points: List[np.ndarray] = self.generateTargetPoints(node)
            goal: Tuple[float, float] = self.scenario.getGoal()

            for target_point in target_points:
                astar_grid_time: float = grid_time_factor * self.heuristic(node, (goal[0], goal[1], 0.0))
                max_steps: int = max(3, int(astar_grid_time / self.time_step))
                self.setTargetPoints(target_points=[target_point])

                simulation_data: Optional[Dict] = self.simulate(
                    sim_id='Astar', state=node, max_steps=max_steps, time_step=self.time_step
                )

                if simulation_data is None:
                    continue

                last_state: Tuple[float, ...] = tuple(simulation_data['states_history'][-1])
                neighbors.append(last_state)
                neighbors_times[last_state] = simulation_data['time_history'][-1]

                from_key: Tuple[float, ...] = tuple(node)
                to_key: Tuple[float, ...] = last_state
                self.current_to_neighbor_controls_history.setdefault(from_key, {})[to_key] = copy.deepcopy(
                    simulation_data['controls_history']
                )
            return neighbors, neighbors_times
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.getNeighbors failed: {e}")
            raise

    def generatePath(
        self, goal_node: Tuple[float, ...], came_from_node_list: Dict[Tuple, Tuple]
    ) -> List[Tuple[float, ...]]:
        """Reconstruct the path from the start node to the goal node.

        Backtracks from the goal using the `came_from_node_list` dictionary
        and assembles the complete control history for the path.

        Parameters
        ----------
        goal_node : Tuple[float, ...]
            The final node reached by the A* search.
        came_from_node_list : Dict[Tuple, Tuple]
            A dictionary mapping each node to its predecessor.

        Returns
        -------
        List[Tuple[float, ...]]
            The reconstructed path as a list of state tuples.

        Raises
        ------
        Exception
            Propagates any error that occurs during path reconstruction.
        """
        try:
            node: Tuple[float, ...] = tuple(float(v) for v in goal_node)
            path: List[Tuple[float, ...]] = [node]
            self.controls_history = []
            self.num_controls_history = 0

            while node in came_from_node_list:
                prev_node: Tuple[float, ...] = tuple(float(v) for v in came_from_node_list[node])
                path.append(prev_node)

                from_key: Tuple[float, ...] = tuple(prev_node)
                to_key: Tuple[float, ...] = tuple(node)
                prev_controls: List = self.current_to_neighbor_controls_history.get(from_key, {}).get(to_key, [])

                if not self.controls_history:
                    self.controls_history = list(prev_controls)
                else:
                    self.controls_history = list(prev_controls)[:-1] + self.controls_history
                node = prev_node

            path.reverse()
            self.num_controls_history = len(self.controls_history)
            return path
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.generatePath failed: {e}")
            raise

    def controlHistory(self, step: int, state: Tuple[float, ...]) -> Tuple[float, float]:
        """Provide control inputs for a given step from the computed path.

        This function acts as a callback for the simulator to replay the
        A*-computed trajectory.

        Parameters
        ----------
        step : int
            The current simulation step index.
        state : Tuple[float, ...]
            The current state (unused in this implementation).

        Returns
        -------
        Tuple[float, float]
            The control tuple (velocity, angular_rate) for the given step.

        Raises
        ------
        Exception
            Propagates any error that occurs during control retrieval.
        """
        try:
            if self.num_controls_history < 1 or self.controls_history is None:
                return (0.0, 0.0)
            return self.controls_history[step] if step < self.num_controls_history else self.controls_history[-1]
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.controlHistory failed: {e}")
            raise

    def astar(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
        max_execution_time: float,
        grid_time_factor: float = 1.0
    ) -> Tuple[Optional[List[Tuple[float, ...]]], int]:
        """Execute the A* search algorithm.

        Parameters
        ----------
        start : Tuple[float, ...]
            The initial state (x, y, theta).
        goal : Tuple[float, ...]
            The final state (xg, yg, theta_g).
        max_execution_time : float
            The maximum allowed time in seconds for the search.
        grid_time_factor : float, optional
            Factor scaling simulation time for neighbor generation, by default 1.0.

        Returns
        -------
        Tuple[Optional[List[Tuple[float, ...]]], int]
            A tuple containing the found path (or None) and the number of
            nodes expanded during the search.

        Raises
        ------
        Exception
            Propagates any error that occurs during the search.
        """
        try:
            open_set: List[Tuple[float, Tuple[float, ...]]] = []
            heapq.heappush(open_set, (0, tuple(start)))

            came_from_node_list: Dict[Tuple, Tuple] = {}
            g_score: Dict[Tuple, float] = {tuple(start): 0}
            f_score: Dict[Tuple, float] = {tuple(start): self.heuristic(start, goal)}

            self.controls_history = None
            astar_num_steps: int = 0
            execution_start_time: float = time.perf_counter()
            goal_radius: float = self.scenario.getGoalRadius()

            while open_set:
                astar_num_steps += 1
                if time.perf_counter() - execution_start_time >= max_execution_time:
                    _, current_node = open_set[0]
                    return self.generatePath(current_node, came_from_node_list), astar_num_steps

                _, current_node = heapq.heappop(open_set)

                if self.distance(current_node[:2], goal[:2]) <= goal_radius:
                    return self.generatePath(current_node, came_from_node_list), astar_num_steps

                neighbors, neighbors_times = self.getNeighbors(current_node, grid_time_factor)

                for neighbor in neighbors:
                    tentative_g_score: float = g_score[current_node] + neighbors_times[neighbor]
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from_node_list[neighbor] = current_node
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

            return None, astar_num_steps
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.astar failed: {e}")
            raise

    def solve(self, max_steps: int, time_step: float, max_execution_time: float) -> Optional[Dict]:
        """Execute the main solver method.

        This method first attempts a fast analytical solution. It then
        iteratively runs the A* algorithm with a progressively refined
        strategy to find the best path within the given execution time.

        Parameters
        ----------
        max_steps : int
            The maximum number of simulation steps allowed for any path.
        time_step : float
            The duration of a single simulation step.
        max_execution_time : float
            The total time budget for the solver.

        Returns
        -------
        Optional[Dict]
            A dictionary with simulation results of the best path found,
            or None if no solution is found.

        Raises
        ------
        Exception
            Propagates any error that occurs during the solving process.
        """
        try:
            self.kp: float = 0.5 / time_step
            self.max_steps: int = max_steps
            self.time_step: float = time_step

            start: Tuple[float, float] = self.scenario.getStart()
            goal: Tuple[float, float] = self.scenario.getGoal()
            self.initial_goal_distance: float = self.distance(start, goal[:2])
            initial_state: Tuple[float, ...] = (float(start[0]), float(start[1]), float(self.scenario.getInitialHeading()))
            final_state: Tuple[float, ...] = (float(goal[0]), float(goal[1]), 0.0)
            execution_start_time: float = time.perf_counter()
            execution_time: float = 0.0
            best_astar_simulation_data: Optional[Dict] = None

            analytic_simulation_data: Optional[Dict] = super().solve(
                max_steps=max_steps, time_step=time_step, max_execution_time=max_execution_time
            )

            if analytic_simulation_data is not None and analytic_simulation_data['goal_objective']:
                best_astar_simulation_data = analytic_simulation_data
                best_astar_simulation_data['id'] = 'Astar'
                self.max_steps = best_astar_simulation_data['num_steps']

            execution_time = time.perf_counter() - execution_start_time
            grid_time_factor: float = 1.0

            while True:
                remaining_time: float = max_execution_time - execution_time
                if remaining_time < 0:
                    break

                astar_num_steps: int
                self.path, astar_num_steps = self.astar(
                    start=initial_state,
                    goal=final_state,
                    max_execution_time=remaining_time,
                    grid_time_factor=grid_time_factor
                )

                if self.path:
                    self.setTargetPoints(target_points=self.path)
                    astar_simulation_data: Optional[Dict] = self.simulate(
                        sim_id='Astar',
                        state=initial_state,
                        max_steps=self.max_steps,
                        time_step=self.time_step,
                        control_function=self.controlHistory
                    )
                    if astar_simulation_data is not None and astar_simulation_data['goal_objective']:
                        astar_simulation_data['astar_num_steps'] = astar_num_steps
                        if best_astar_simulation_data is not None:
                            best_astar_simulation_data = self.chooseBestSolution(
                                best_astar_simulation_data, astar_simulation_data
                            )
                        else:
                            best_astar_simulation_data = astar_simulation_data
                        
                        best_astar_simulation_data['grid_time_factor'] = grid_time_factor
                        self.max_steps = best_astar_simulation_data['num_steps']

                grid_time_factor = round(grid_time_factor - 0.1, 2)
                if grid_time_factor < 0.1:
                    break

                execution_time = time.perf_counter() - execution_start_time

            return best_astar_simulation_data
        except Exception as e:
            print(f"[ERROR] SolverZermeloAStar.solve failed: {e}")
            raise