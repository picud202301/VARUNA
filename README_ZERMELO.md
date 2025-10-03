# Zermelo Problem Benchmark

[![Module](https://img.shields.io/badge/module-zermelo-blue)]()
[![Status](https://img.shields.io/badge/status-baseline-success)]()

This module provides a **standardized benchmark** for the classical **Zermelo navigation problem**, enabling consistent, reproducible comparisons across heterogeneous solution methods.

## Problem (brief)
Given a 2D flow field **u(x, y)** and a vessel with constant speed **V**, find a heading (or heading-rate) control that **minimizes travel time** between a start and a goal while respecting domain bounds (and optional constraints).

> Associated paper: *Standardizing Navigation Algorithms: A Benchmarking Framework for the Zermelo Problem*.


## Quick Start


1) **Run the Zermelo baseline** (single instance; compare active solvers)

    python run_problem.py

    # Config: code/problems/problems.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: 'zermelo'
    # SCENARIO_TYPE: 'random'   # {'fixed', 'random'}
    # CURRENT_TYPE: Optional[str] = None  # None => sample from:
    #   ["uniform","sinusoidal","logarithmic","gaussianSwirl","vortex",
    #    "karmanVortex","coastalTidal","linearShear","doubleGyre",
    #    "gaussianJet","riverOutflow","turbulenceNoise"]
    # SIZE_ID: 1  # {1:200x200, 2:2000x2000, 3:20000x20000}
    # MASTER_SEED: Optional[int] = None  # set int for reproducibility

2) **Run batch simulations** (build the benchmark database)

    python run_simulations.py

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

3) **Generate reports** (tables/figures for manuscripts)

    python report_simulations.py

---

## Extending Solvers (Zermelo)

1) **Create a solver module**
    
    problems/zermelo/solvers/<your_solver_name>/
      ├── __init__.py
      └── solver.py

2) **Register the solver**
    
    problems/zermelo/problems.py
    # Add to AVAILABLE_SOLVERS:
    #   - "solve": callable
    #   - "meta":  dict (name, deterministic, params)

3) **Implement the minimal interface**

    # problems/zermelo/solvers/<your_solver_name>/solver.py
    from typing import Dict, Any
    import numpy as np

    METADATA: Dict[str, Any] = {
        "name": "YourSolver",
        "deterministic": True,   # or False
        "params": {"max_steps": 10_000, "time_step": 0.1},
    }

    def control(t: float, x: np.ndarray, problem, **kwargs):
        # Return control (e.g., heading or rate).
        return 0.0  # placeholder

    def solve(problem, **kwargs):
        # Use problem.simulate(...) to standardize outputs.
        return problem.simulate(
            sim_id=METADATA["name"],
            state=problem.initial_state,
            max_steps=kwargs.get("max_steps", METADATA["params"]["max_steps"]),
            time_step=kwargs.get("time_step", METADATA["params"]["time_step"]),
            max_execution_time=kwargs.get("max_execution_time", None),
            controller=lambda t, x: control(t, x, problem, **kwargs),
        )

4) **Required outputs** (for fair comparisons)

    # Must return (directly or via simulate):
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
