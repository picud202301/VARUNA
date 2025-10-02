# ===============================================================================
# Zermelo Navigation Problem plotter of simulations results
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Strongly typed plotting utilities for Zermelo navigation scenarios and solver
#   outputs. This class renders:
#     - The scenario map with the current field (quiver) and solver trajectories.
#     - Time series for heading (deg), heading rate (deg/s), and effective velocities.
#   It expects solver-produced dictionaries with histories and summary metrics
#   (e.g., total_time, total_distance, execution_time, navigation index).
#   Public methods include adding solutions, plotting the scenario, saving, and showing.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from plotters.Plotter import Plotter


class PlotterZermelo(Plotter):
    """
    Plotting utility for Zermelo navigation experiment results. It aggregates
    solver outputs, draws the scenario map with the ambient current field, and
    overlays solver trajectories and time-series diagnostics.

    Attributes
    ----------
    solutions_data : dict[str, dict]
        Mapping from solution id to its simulation data dictionary.
    solutions_solvers : dict[str, Any]
        Mapping from solution id to the solver instance that produced it.
    max_time : float
        Maximum total time across all registered solutions (used for x-limits).
    goal_fontsize : float
        Font size used for goal annotation text.
    axis_fontsize : float
        Font size for axis labels and titles.
    legend_fontsize : float
        Font size for legend text.
    line_width : float
        Line width for plotted lines.
    grid_alpha : float
        Grid transparency for auxiliary grids.
    goal_edgecolor : str
        Edge color for the goal-radius circle.
    goal_linestyle : str
        Line style for the goal-radius circle.
    goal_marker : str
        Marker style for the goal center.
    _last_figure_id : int | None
        Identifier of the most recently created figure in ``self.figures``.
    """

    def __init__(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the plotter with styling and container structures.

        Parameters
        ----------
        parameters : dict
            Plot configuration; recognized keys (all optional):
              - 'current_quiver_scale' : float (default 50)
              - 'current_quiver_width' : float (default 0.006)
              - 'current_quiver_num_vectors' : int (default 20)
              - 'line_width' : float (default 2.0)
              - 'grid_alpha' : float (default 0.25)
              - 'goal_edgecolor' : str (default 'red')
              - 'goal_linestyle' : str (default '--')
              - 'goal_marker' : str (default 'x')
              - 'tight_layout' : bool (default True)

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization.
        """
        try:
            super().__init__(parameters)
            self.solutions_data: Dict[str, Dict[str, Any]] = {}
            self.solutions_solvers: Dict[str, Any] = {}
            self.max_time: float = 0.0

            # Typography defaults
            self.goal_fontsize: float = 17.0
            self.axis_fontsize: float = 13.0
            self.legend_fontsize: float = 12.0

            # Style defaults (can be overridden by parameters)
            self.line_width: float = float(self.parameters.get("line_width", 2.0))
            self.grid_alpha: float = float(self.parameters.get("grid_alpha", 0.25))
            self.goal_edgecolor: str = str(self.parameters.get("goal_edgecolor", "red"))
            self.goal_linestyle: str = str(self.parameters.get("goal_linestyle", "--"))
            self.goal_marker: str = str(self.parameters.get("goal_marker", "x"))

            self._last_figure_id: int | None = None
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.__init__ failed: {e}")
            raise

    # --------------------------------- helpers ---------------------------------

    @staticmethod
    def _wrapAngleRadians(theta_array: np.ndarray | list[float]) -> np.ndarray:
        """
        Wrap a sequence of angles to the principal branch (-π, π).

        Parameters
        ----------
        theta_array : array-like
            Angles in radians (any range).

        Returns
        -------
        np.ndarray
            Wrapped angles in radians within (-π, π).

        Raises
        ------
        Exception
            Propagates any error during array conversion or arithmetic.
        """
        try:
            arr = np.asarray(theta_array, dtype=float)
            return (arr + np.pi) % (2.0 * np.pi) - np.pi
        except Exception as e:
            print(f"[ERROR] PlotterZermelo._wrapAngleRadians failed: {e}")
            raise

    # --------------------------------- public ----------------------------------

    def addSolution(self, simulation_data: Dict[str, Any], solver: Any) -> None:
        """
        Register a solver's plotted solution.

        Parameters
        ----------
        simulation_data : dict
            Output dictionary produced by a solver. Expected keys include:
              - 'id' : str
              - 'time_history' : array-like
              - 'states_history' : list/array of (x, y, heading)
              - 'controls_history' : list/array of (v, r)
              - 'state_derivatives_history' : list/array of (dx, dy, dtheta)
              - 'total_time', 'total_distance', 'execution_time' : float
              - 'navegation_index' : float
              - 'color' : str
        solver : Any
            Solver instance (used to delegate optional overlays via ``solver.plot(self)``).

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error if required keys are missing or assignment fails.
        """
        try:
            sid: str = str(simulation_data["id"])
            self.solutions_data[sid] = simulation_data
            self.solutions_solvers[sid] = solver
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.addSolution failed: {e}")
            raise

    def plotCurrentField(self, scenario: Any) -> None:
        """
        Plot the current field as a quiver on the "scenario" axis.

        Parameters
        ----------
        scenario : object
            Scenario object with:
              - getSizeX(), getSizeY()
              - getCurrentField().getVelocity(), getCurrentField().getCurrentAtPosition(x, y)

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error from current-field queries or plotting.
        """
        try:
            axis_id: str = "scenario"
            current_quiver_scale: float = float(self.parameters.get("current_quiver_scale", 50))
            current_quiver_width: float = float(self.parameters.get("current_quiver_width", 0.006))
            current_quiver_num_vectors: int = int(self.parameters.get("current_quiver_num_vectors", 20))

            scenario_size_x: float = float(scenario.getSizeX())
            scenario_size_y: float = float(scenario.getSizeY())
            _ = scenario.getCurrentField().getVelocity()

            x_coords: np.ndarray = np.linspace(-scenario_size_x, scenario_size_x, current_quiver_num_vectors)
            y_coords: np.ndarray = np.linspace(-scenario_size_y, scenario_size_y, current_quiver_num_vectors)
            X, Y = np.meshgrid(x_coords, y_coords)

            U: np.ndarray = np.zeros_like(X, dtype=float)
            V: np.ndarray = np.zeros_like(Y, dtype=float)

            current_field: Any = scenario.getCurrentField()
            for i in range(current_quiver_num_vectors):
                for j in range(current_quiver_num_vectors):
                    vec = np.asarray(current_field.getCurrentAtPosition(float(X[i, j]), float(Y[i, j])), dtype=float).ravel()
                    if vec.size < 2:
                        u_val, v_val = 0.0, 0.0
                    else:
                        u_val, v_val = float(vec[0]), float(vec[1])
                    U[i, j] = u_val
                    V[i, j] = v_val

            self.axis[axis_id].quiver(
                X, Y, U, V,
                scale=current_quiver_scale,
                width=current_quiver_width,
                color="lightslategray",
                alpha=0.8,
            )
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.plotCurrentField failed: {e}")
            raise

    def plotSolverTimeData(self, scenario: Any) -> None:
        """
        Plot time-based series: heading (deg), heading rate (deg/s), and effective velocities.

        Parameters
        ----------
        scenario : object
            Scenario object (unused here but kept for API symmetry).

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during data extraction or plotting.
        """
        try:
            heading_axis_id: str = "heading"
            heading_rate_axis_id: str = "heading_rate"
            effective_velocity_x_axis_id: str = "effective_velocity_x"
            effective_velocity_y_axis_id: str = "effective_velocity_y"

            for _solution_id, simulation_data in self.solutions_data.items():
                time_history: np.ndarray = np.asarray(simulation_data["time_history"], dtype=float)

                states_history = simulation_data["states_history"]
                _x_arr, _y_arr, heading_arr = zip(*states_history)
                heading_wrapped: np.ndarray = self._wrapAngleRadians(np.array(heading_arr, dtype=float))
                heading_deg: np.ndarray = np.rad2deg(heading_wrapped)

                controls_history = simulation_data["controls_history"]
                _v_arr, r_arr = zip(*controls_history)
                heading_rate_deg: np.ndarray = np.rad2deg(np.array(r_arr, dtype=float))

                state_derivatives_history = simulation_data["state_derivatives_history"]
                dx_arr, dy_arr, _ = zip(*state_derivatives_history)

                label: str = (
                    f"{simulation_data['id']}: "
                    f"T={simulation_data['total_time']:.2f} s, "
                    f"D={simulation_data['total_distance']:.2f} m, "
                    f"ET={simulation_data['execution_time']:.2f} s"
                )

                self.axis[heading_axis_id].plot(
                    time_history,
                    heading_deg,
                    label=label,
                    color=simulation_data.get("color", "grey"),
                    linewidth=self.line_width,
                )
                self.axis[heading_rate_axis_id].plot(
                    time_history,
                    heading_rate_deg,
                    label=label,
                    color=simulation_data.get("color", "grey"),
                    linewidth=self.line_width,
                )
                self.axis[effective_velocity_x_axis_id].plot(
                    time_history,
                    np.asarray(dx_arr, dtype=float),
                    label=label,
                    color=simulation_data.get("color", "grey"),
                    linewidth=self.line_width,
                )
                self.axis[effective_velocity_y_axis_id].plot(
                    time_history,
                    np.asarray(dy_arr, dtype=float),
                    label=label,
                    color=simulation_data.get("color", "grey"),
                    linewidth=self.line_width,
                )
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.plotSolverTimeData failed: {e}")
            raise

    def plotSolverTrajectories(self, scenario: Any) -> None:
        """
        Plot solver 2D trajectories on the "scenario" axis and allow solver overlays.

        Parameters
        ----------
        scenario : object
            Scenario object (used to query goal or limits indirectly if needed).

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during trajectory extraction or plotting.
        """
        try:
            axis_id: str = "scenario"

            # Select best solution by navigation index (lower is better)
            best_solution_id: str | None = None
            best_navigation_index: float = float("inf")
            for sid, sdata in self.solutions_data.items():
                if sdata["navegation_index"] < best_navigation_index:
                    best_navigation_index = sdata["navegation_index"]
                    best_solution_id = sid

            for solution_id, simulation_data in self.solutions_data.items():
                states_history = simulation_data["states_history"]
                x_history, y_history, _ = zip(*states_history)

                label: str = (
                    f"{simulation_data['id']}: "
                    f"T={simulation_data['total_time']:.2f} s, "
                    f"D={simulation_data['total_distance']:.2f} m, "
                    f"ET={simulation_data['execution_time']:.2f} s, "
                    f"NI={simulation_data['navegation_index']:.2f}"
                )

                line, = self.axis[axis_id].plot(
                    x_history,
                    y_history,
                    label=label,
                    color=simulation_data.get("color", "grey"),
                    linewidth=self.line_width,
                )

                # Highlight best solution
                if solution_id == best_solution_id:
                    line.set_linewidth(self.line_width * 1.5)
                    line.set_zorder(10)
                    leg_font = fm.FontProperties(weight="bold")
                    line.set_label(label + " ★")
                    self.axis[axis_id].legend(prop=leg_font)

                # Allow solver to draw overlays
                solver: Any | None = self.solutions_solvers.get(solution_id, None)
                if solver is not None and hasattr(solver, "plot"):
                    try:
                        solver.plot(self)
                    except Exception:
                        # If solver overlay fails, continue rendering base plots
                        pass

                # Track global max time
                self.max_time = max(self.max_time, float(simulation_data.get("total_time", 0.0)))
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.plotSolverTrajectories failed: {e}")
            raise

    def plot(self, scenario: Any) -> None:
        """
        Create the full figure, including scenario map, heading, heading rate,
        and effective velocities.

        Parameters
        ----------
        scenario : object
            Scenario object exposing:
              - getSizeX(), getSizeY()
              - getGoal(), getGoalRadius()
              - getCurrentField().getCurrentType()

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during figure creation or plotting.
        """
        try:
            figure_id: int = 1
            self.createFigure(figure_id=figure_id, figure_size=(16, 8))
            self._last_figure_id = figure_id

            gs: GridSpec = GridSpec(3, 3, figure=self.figures[figure_id], width_ratios=[0.5, 0.25, 0.25])

            # ---------------------- SCENARIO AXIS ----------------------
            axis_id: str = "scenario"
            self.axis[axis_id] = self.figures[figure_id].add_subplot(gs[:, 0])

            self.plotCurrentField(scenario=scenario)
            self.plotSolverTrajectories(scenario=scenario)

            # Goal (disk + marker + text)
            current_field: Any = scenario.getCurrentField()
            current_type: Any = current_field.getCurrentType()
            goal: tuple[float, float] = scenario.getGoal()
            goal_radius: float = float(scenario.getGoalRadius())
            goal_circle = plt.Circle(goal, goal_radius, fill=False, edgecolor=self.goal_edgecolor, linestyle=self.goal_linestyle)
            self.axis[axis_id].add_patch(goal_circle)
            self.axis[axis_id].plot(goal[0], goal[1], self.goal_marker, color=self.goal_edgecolor, markersize=6)
            self.axis[axis_id].text(
                goal[0] - 1.5 * goal_radius,
                goal[1] - 1.5 * goal_radius,
                f"({goal[0]:.2f}, {goal[1]:.2f})",
                fontsize=self.goal_fontsize,
                color="black",
            )

            scenario_size_x: float = float(scenario.getSizeX())
            scenario_size_y: float = float(scenario.getSizeY())
            self.axis[axis_id].set_xlim(-scenario_size_x, scenario_size_x)
            self.axis[axis_id].set_ylim(-scenario_size_y, scenario_size_y)
            self.axis[axis_id].set_aspect("equal", adjustable="box")

            self.axis[axis_id].set_xlabel("X (m)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_ylabel("Y (m)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_title(
                f"Scenario: ({scenario_size_x:.1f}, {scenario_size_y:.1f}) m; Current Field: {current_type}",
                fontsize=self.axis_fontsize,
            )
            self.axis[axis_id].grid(True, alpha=self.grid_alpha)

            _handles, labels = self.axis[axis_id].get_legend_handles_labels()
            if labels:
                self.axis[axis_id].legend(fontsize=self.legend_fontsize, loc="lower right")

            # ---------------------- HEADING AXIS ----------------------
            axis_id = "heading"
            self.axis[axis_id] = self.figures[figure_id].add_subplot(gs[0, 1:3])
            if self.max_time > 0:
                self.axis[axis_id].set_xlim(0.0, self.max_time)
            self.axis[axis_id].set_ylim(-180.0, 180.0)
            self.axis[axis_id].set_xlabel("Time (s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_ylabel("Heading (°)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_title("Heading", fontsize=self.axis_fontsize)
            self.axis[axis_id].grid(True, alpha=self.grid_alpha)
            self.axis[axis_id].set_yticks([-180, -90, 0, 90, 180])

            # ---------------------- HEADING RATE AXIS ----------------------
            axis_id = "heading_rate"
            self.axis[axis_id] = self.figures[figure_id].add_subplot(gs[1, 1:3])
            if self.max_time > 0:
                self.axis[axis_id].set_xlim(0.0, self.max_time)
            self.axis[axis_id].set_xlabel("Time (s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_ylabel("Heading rate (°/s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_title("Heading rate control", fontsize=self.axis_fontsize)
            self.axis[axis_id].grid(True, alpha=self.grid_alpha)

            # ---------------------- EFFECTIVE VELOCITY X ----------------------
            axis_id = "effective_velocity_x"
            self.axis[axis_id] = self.figures[figure_id].add_subplot(gs[2, 1])
            if self.max_time > 0:
                self.axis[axis_id].set_xlim(0.0, self.max_time)
            self.axis[axis_id].set_xlabel("Time (s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_ylabel("Velocity (m/s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_title("Effective velocity in x", fontsize=self.axis_fontsize)
            self.axis[axis_id].grid(True, alpha=self.grid_alpha)

            # ---------------------- EFFECTIVE VELOCITY Y ----------------------
            axis_id = "effective_velocity_y"
            self.axis[axis_id] = self.figures[figure_id].add_subplot(gs[2, 2])
            if self.max_time > 0:
                self.axis[axis_id].set_xlim(0.0, self.max_time)
            self.axis[axis_id].set_xlabel("Time (s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_ylabel("Velocity (m/s)", fontsize=self.axis_fontsize)
            self.axis[axis_id].set_title("Effective velocity in y", fontsize=self.axis_fontsize)
            self.axis[axis_id].grid(True, alpha=self.grid_alpha)

            # Render time-series
            self.plotSolverTimeData(scenario=scenario)

            if bool(self.parameters.get("tight_layout", True)):
                plt.tight_layout()
        except Exception as e:
            print(f"[ERROR] PlotterZermelo.plot failed: {e}")
            raise
