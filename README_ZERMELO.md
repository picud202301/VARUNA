# Zermelo Problem Benchmark

[![Module](https://img.shields.io/badge/module-zermelo-blue)]()
[![Status](https://img.shields.io/badge/status-baseline-success)]()

This module provides a **standardized benchmark** for the classical **Zermelo navigation problem**, enabling consistent, reproducible comparisons across heterogeneous solution methods.

## Problem (brief)
Given a 2D flow field **u(x, y)** and a vessel with constant speed **V**, find a heading (or heading-rate) control that **minimizes travel time** between a start and a goal while respecting domain bounds (and optional constraints).

> Associated paper: *Standardizing Navigation Algorithms: A Benchmarking Framework for the Zermelo Problem*.

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

    bash
    python run_problem.py

The execution is defined with the following elements:

    # =======================================================================
    # CONFIGURE EXECUTION
    # =======================================================================
    PROBLEM_NAME: str = 'zermelo' 
    SCENARIO_TYPE: str = 'random'  # Options: 'fixed', 'random'
    CURRENT_TYPE: Optional[str] = None  # If None, randomly selected from:
    # ["uniform", "sinusoidal", "logarithmic", "gaussianSwirl", "vortex",
    #  "karmanVortex", "coastalTidal", "linearShear", "doubleGyre",
    #  "gaussianJet", "riverOutflow", "turbulenceNoise"]
    SIZE_ID: int = 1  # Options: 1:(200x200)m; 2:(2000x2000)m; 3:(20000x20000)m
    MASTER_SEED: Optional[int] = None  # If None, a random seed is used. Set an integer to generate reproducible repetitions.

### Run multiple simulations to generate benchmark database data 

    bash
    python run_simulations.py

The execution is defined with the following elements:

    PROBLEM_NAME: str = "zermelo"
    SCENARIO_TYPE: str = "random"  # If 'random', a new scenario is executed in every simulation. If 'fixed', all simulations share the same conditions except for the goal point.
    MASTER_SEED: int = 1  # Global seed to ensure reproducible results. If None, a random seed is used.
    NUM_SIMULATIONS: int = 1000  # Number of simulations for each scenario and current type.
    SIZES_ID: Optional[Iterable[int]] = None  # e.g., [1]  # List of size IDs to be evaluated. If None, all sizes are used. Default [1, 2, 3]
    CURRENTS_ID: Optional[Iterable[int]] = None  # List of currents to be evaluated (by index). If None, all current types are used. Default [0, .., 11]
    DATABASE_FILE: str = os.path.abspath(os.path.join(DATA_PATH, "zermelo", "zermelo.db"))  # Path to the database file
    DB_RESET: bool = True  # If True, the database is cleared before running simulations.
    PARALLEL_EXECUTION: bool = True  # Enable parallel execution of scenarios to speed up results.

### Generate graphics and reporting information (tables/figures for the manuscript)

    bash
    python report_simulations.py

---

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
