# Zermelo Problem Benchmark

[![Module](https://img.shields.io/badge/module-zermelo-blue)]()
[![Status](https://img.shields.io/badge/status-baseline-success)]()

This module provides a **standardized benchmark** for the classical **Zermelo navigation problem**, enabling consistent, reproducible comparisons across heterogeneous solution methods.

## Problem (brief)
Given a 2D flow field **u(x, y)** and a vessel with constant speed **V**, find a heading (or heading-rate) control that **minimizes travel time** between a start and a goal while respecting domain bounds (and optional constraints).

> Associated paper: *Standardizing Navigation Algorithms: A Benchmarking Framework for the Zermelo Problem*.


## Quick Start
Set up  your folder as VARUNA/code/.

1) **Run the Zermelo baseline** (single instance; compare active solvers)

    python run_problem.py
   <pre style="font-size:12px">
    # Configured with the following  parameters:
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: 'zermelo'
    # SCENARIO_TYPE: 'random'   # Options: {'fixed', 'random'}. fixed is the scenario used in the analysis  section of the manuscript.
    # CURRENT_TYPE: Optional[str] = None  # None => sample random from:
    #   ["uniform","sinusoidal","logarithmic","gaussianSwirl","vortex",
    #    "karmanVortex","coastalTidal","linearShear","doubleGyre",
    #    "gaussianJet","riverOutflow","turbulenceNoise"]
    # SIZE_ID: 1  # {1:200x200, 2:2000x2000, 3:20000x20000}
    # MASTER_SEED: Optional[int] = None  # set int for reproducibility
   </pre>
2) **Run batch simulations** (build the benchmark database)

    python run_simulations.py
   <pre style="font-size:12px">
    # Config: code/run_simulations.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: "zermelo"
    # SCENARIO_TYPE: "random"  # new scenario per sim if 'random'
    # MASTER_SEED: Optional[int] = 1  # None => random
    # NUM_SIMULATIONS: 1000      # per scenario & current
    #
    # SIZES_ID: Optional[Iterable[int]] = None   # e.g., [1]; None => all
    # CURRENTS_ID: Optional[Iterable[int]] = None  # None => all (0..11)
    #
    # DATABASE_FILE: os.path.abspath(os.path.join(DATA_PATH,"zermelo","zermelo.db"))
    # DB_RESET: True            # clear DB before running
    # PARALLEL_EXECUTION: True  # enable parallel sims
   </pre>
3) **Generate reports** (tables/figures for manuscripts)

    python report_simulations.py

---

## Extending Solvers (Zermelo)

1) **Create a solver module**
    Create a solver class named SolverZermeloYourSolver in: problems/zermelo/solvers/SolverZermeloYourSolver.py
    

2) **Register the solver**
    Resgister a your solver in file:  problems/zermelo/problems.py as follows:
    <pre style="font-size:12px">
    # ---------------------- Configure solvers ----------------------
      from problems.zermelo.solvers.SolverZermeloAnalytic import SolverZermeloAnalytic
      from problems.zermelo.solvers.SolverZermeloIpopt import SolverZermeloIpopt
      from problems.zermelo.solvers.SolverZermeloAStar import SolverZermeloAStar
      from problems.zermelo.solvers.SolverZermeloPSO import SolverZermeloPSO
      from problems.zermelo.solvers.SolverZermeloYourSolver import SolverZermeloYourSolver
      from problems.zermelo.ProblemZermelo import ProblemZermelo
      solvers_configuration = {
         'analytic':   {'class': SolverZermeloAnalytic, 'active': True, 'parameters':{'color':'red',  'library': 'np'}},
         'astar':      {'class': SolverZermeloAStar,     'active': True, 'parameters':{'color':'blue',  'library': 'np'}},
         'pso':        {'class': SolverZermeloPSO,       'active': True, 'parameters':{'color':'black',  'library': 'np'}},
         'ipopt':      {'class': SolverZermeloIpopt,    'active': True, 'parameters':{'color':'green',  'library': 'pyo'}},
         'your_solver': {'class': SolverZermeloYourSolver,       'active': True, 'parameters':{'color':'magenta',  'library': 'np'}},
      }
      </pre>
3) **Implement the solver interface**
    In class SolverZermeloYourSolver there must be the following 2 functions:
    <pre style="font-size:12px">
    def control(self, step: int, state: List[float]) -> np.ndarray:
        # Your control code: generate ship_velocity and heading_rate
        ship_velocity: float = self.scenario.getShipVelocity()
        heading_rate : float = <your control solution>
        heading_rate = np.clip(heading_rate, self.scenario.getRMin(), self.scenario.getRMax())
        return np.array([ship_velocity, heading_rate], dtype=float)

    def solve(self, max_steps: int, time_step: float, max_execution_time: float) -> Optional[Dict]:
        # Your code to solve the problem and configure the control function response
        simulation_data: Optional[Dict] = self.simulate(
                sim_id="Your solver",
                state=initial_state,
                max_steps=max_steps,
                time_step=time_step,
                max_execution_time=max_execution_time
            )
            return simulation_data
        
   </pre>
4) **Required outputs** (for fair comparisons)

    # Must return (directly or via simulate):
    <pre style="font-size:12px">
    simulation_data = {
        "id": str,
        "time_step": float,
        "num_steps": int,
        "goal_objective": (float, float),
        "total_time": float,
        "total_distance": float,
        "last_state": np.ndarray,
        "distance_to_goal": float,
        "time_history": np.ndarray,
        "states_history": np.ndarray,
        "controls_history": np.ndarray,
        "disturbance_history": np.ndarray,
        "state_derivatives_history": np.ndarray,
        "navigation_index": float
    }
   </pre>
---

## Reproducibility Notes

- Use a fixed `MASTER_SEED` for pseudo-randomness in scenarios and (if applicable) solvers.
- Keep `SCENARIO_TYPE`, `SIZE_ID`, and `CURRENT_TYPE` consistent when comparing methods.
- All solvers run under the same protocol (identical scenarios, step sizes, stopping criteria) to ensure like-for-like, repeatable evaluations.

---


---

## Implemented Solvers
Representative strategies included in the benchmark design:
1. **Analytical** solution for **uniform flows**.
2. **A\*-based search** with **analytical guidance**.
3. **Nonlinear optimization** (e.g., with **IPOPT**).
4. **Particle Swarm Optimization (PSO)**.

This mix of deterministic, continuous, and heuristic methods supports comparison on **travel time**, **path length**, and **execution time**.

---

## Entry Points & Usage (current layout with `code/`)
If your repository keeps runners under `code/`, use:

### Run a single Zermelo instance and compare results from different solvers
_Solver configuration and activation can be modified in_ `code/problems/problems.py`.


## Extending Zermelo Solvers
You can add new solution methods by **creating a solver** and **activating** it:

1. **Create** a new solver module under:

        problems/zermelo/solvers/

2. **Activate/register** the new solver in:

        problems/zermelo/problems.py

3. **Interface guidelines** (suggested):
   - Provide a `solve(problem, **kwargs) -> Solution` entry point.
   - Include metadata (name, deterministic/stochastic, required parameters).
   - Export standard metrics so `run_simulations.py` and `report_simulations.py` can compare results consistently.

---

## Tips
- Keep scenarios and seeds fixed for reproducibility.
- For fair comparisons, every solver should return the following data structure:

        simulation_data = {
                            "id": str,
                            "time_step": float,
                            "num_steps": int,
                            "goal_objective": (float, float),
                            "total_time": float,
                            "total_distance": float,
                            "last_state": np.ndarray,
                            "distance_to_goal": float,
                            "time_history": np.ndarray,
                            "states_history":  np.ndarray,
                            "controls_history":  np.ndarray,
                            "disturbance_history":  np.ndarray,
                            "state_derivatives_history": np.ndarray,
                            "navigation_index": float
                        }

To obtain this dictionary, each solver typically exposes a `solve(...)` function which, at the end, invokes the framework’s `simulate(...)`. The simulator will call your `control(...)` policy and assemble standardized outputs:

        simulation_data = self.simulate(
                        sim_id="Your solver",
                        state=initial_state,
                        max_steps=max_steps,
                        time_step=time_step,
                        max_execution_time=max_execution_time
                    )
