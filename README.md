
<p align="center">
  <img src="logo_varuna.png" alt="NARUVA / VARUNA Logo" width="220">
</p>

# NARUVA: A Modular Benchmarking Framework for Marine Navigation

[![Status](https://img.shields.io/badge/status-alpha-informational)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

> Standardized, extensible experiments for marine navigation under ocean currents, starting from the classical **Zermelo** problem.

## Overview
Autonomous marine navigation spans route planning in dynamic environments, compliance with COLREGs, obstacle avoidance, multi-vehicle coordination, and decision-making under uncertainty. Environmental disturbances—currents, wind, and waves—and the trade-offs among travel time, safety, energy, and robustness highlight the need for **standardized and reproducible** benchmarks.  
**NARUVA** is an open-source, **modular** and **extensible** framework that starts from the Zermelo problem and scales to richer settings. It is explicitly oriented toward reproducible results and easy extensibility: datasets and configurations are standardized for like-for-like comparisons, and a clear solver interface allows optimization- and learning-based methods to be added as plug-ins with minimal effort. All solvers run under a common benchmark protocol, enabling fair, repeatable evaluations.

---

## Supported Problems

- **Zermelo (baseline)** — Time-optimal navigation at constant speed within a flow field, with a standardized setup enabling fair, repeatable comparisons of solution methods.  
  → See **[README_ZERMELO.md](./README_ZERMELO.md)** for detailed problem definition, implemented solvers, and usage examples.

- **Planning (Zermelo extensions)** — *Under construction*  
  Roadmap items include realistic ship geometries, time-varying/data-driven current fields, obstacle representations, adaptive time-stepping, coupled ship/wind/wave dynamics with trajectory tracking, and energy-aware metrics.

---

## Quick Start

1) **Clone**
    
    git clone https://github.com/picud202301/NARUVA.git
    cd NARUVA

2) **Set up the environment** (pick one)

    # Using uv/pip (example)
    python -m venv .venv
    source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

    # Or as editable install (if provided)
    pip install -e .

3) **Run the Zermelo baseline** (single instance; compare active solvers)

    python run_problem.py

    # Configure in: code/problems/problems.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: str = 'zermelo'
    # SCENARIO_TYPE: str = 'random'   # {'fixed', 'random'}
    # CURRENT_TYPE: Optional[str] = None  # None => sampled from:
    #   ["uniform", "sinusoidal", "logarithmic", "gaussianSwirl", "vortex",
    #    "karmanVortex", "coastalTidal", "linearShear", "doubleGyre",
    #    "gaussianJet", "riverOutflow", "turbulenceNoise"]
    # SIZE_ID: int = 1  # {1: 200x200 m, 2: 2000x2000 m, 3: 20000x20000 m}
    # MASTER_SEED: Optional[int] = None  # set an int for reproducibility

4) **Run batch simulations** (build the benchmark database)

    python run_simulations.py

    # Configure in: code/run_simulations.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: str = "zermelo"
    # SCENARIO_TYPE: str = "random"  # 'random' => new scenario per simulation
    # MASTER_SEED: Optional[int] = 1 # global seed; None => random
    # NUM_SIMULATIONS: int = 1000    # per scenario and current type
    #
    # SIZES_ID: Optional[Iterable[int]] = None   # e.g., [1]; None => all sizes
    # CURRENTS_ID: Optional[Iterable[int]] = None  # None => all (0..11)
    #
    # DATABASE_FILE: str = os.path.abspath(os.path.join(
    #     DATA_PATH, "zermelo", "zermelo.db"))
    # DB_RESET: bool = True           # True => clear DB before running
    # PARALLEL_EXECUTION: bool = True # enable parallel execution

5) **Generate reports** (tables/figures for manuscripts)

    python report_simulations.py

---

## Extending Solvers (Zermelo)

1) **Create a solver module**
    
    problems/zermelo/solvers/<your_solver_name>/
      ├── __init__.py
      └── solver.py

2) **Register the solver**
    
    problems/zermelo/problems.py
    # Add to AVAILABLE_SOLVERS with:
    #   - "solve": callable entry point
    #   - "meta":  metadata dict (name, deterministic, params)

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
        # Return control (e.g., heading or heading-rate) at time t, state x.
        return 0.0  # placeholder

    def solve(problem, **kwargs):
        # Prefer using problem.simulate(...) to standardize outputs.
        return problem.simulate(
            sim_id=METADATA["name"],
            state=problem.initial_state,
            max_steps=kwargs.get("max_steps", METADATA["params"]["max_steps"]),
            time_step=kwargs.get("time_step", METADATA["params"]["time_step"]),
            max_execution_time=kwargs.get("max_execution_time", None),
            controller=lambda t, x: control(t, x, problem, **kwargs),
        )

4) **Required outputs** (for fair comparisons)

    # Each solver must yield (directly or via simulate) the following:
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

- Use a fixed `MASTER_SEED` to control pseudo-randomness across scenario generation and (where applicable) solver internals.
- Keep `SCENARIO_TYPE`, `SIZE_ID`, and `CURRENT_TYPE` consistent when comparing methods.
- All solvers are executed under the same protocol (identical scenarios, step sizes, and stopping criteria) to ensure like-for-like, repeatable evaluations.

---
<p align="center">
  <img src="logo_varuna.png" alt="NARUVA / VARUNA Logo" width="220">
</p>

# NARUVA: A Modular Benchmarking Framework for Marine Navigation

[![Status](https://img.shields.io/badge/status-alpha-informational)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

> Standardized, extensible experiments for marine navigation under ocean currents, starting from the classical **Zermelo** problem.

## Overview
Autonomous marine navigation spans route planning in dynamic environments, compliance with COLREGs, obstacle avoidance, multi-vehicle coordination, and decision-making under uncertainty. Environmental disturbances—currents, wind, and waves—and the trade-offs among travel time, safety, energy, and robustness highlight the need for **standardized and reproducible** benchmarks.  
**NARUVA** is an open-source, **modular** and **extensible** framework that starts from the Zermelo problem and scales to richer settings. It is explicitly oriented toward reproducible results and easy extensibility: datasets and configurations are standardized for like-for-like comparisons, and a clear solver interface allows optimization- and learning-based methods to be added as plug-ins with minimal effort. All solvers run under a common benchmark protocol, enabling fair, repeatable evaluations.

---

## Supported Problems

- **Zermelo (baseline)** — Time-optimal navigation at constant speed within a flow field, with a standardized setup enabling fair, repeatable comparisons of solution methods.  
  → See **[README_ZERMELO.md](./README_ZERMELO.md)** for detailed problem definition, implemented solvers, and usage examples.

- **Planning (Zermelo extensions)** — *Under construction*  
  Roadmap items include realistic ship geometries, time-varying/data-driven current fields, obstacle representations, adaptive time-stepping, coupled ship/wind/wave dynamics with trajectory tracking, and energy-aware metrics.

---

## Quick Start

1) **Clone**
    
    git clone https://github.com/picud202301/NARUVA.git
    cd NARUVA

2) **Set up the environment** (pick one)

    # Using uv/pip (example)
    python -m venv .venv
    source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

    # Or as editable install (if provided)
    pip install -e .

3) **Run the Zermelo baseline** (single instance; compare active solvers)

    python run_problem.py

    # Configure in: code/problems/problems.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: str = 'zermelo'
    # SCENARIO_TYPE: str = 'random'   # {'fixed', 'random'}
    # CURRENT_TYPE: Optional[str] = None  # None => sampled from:
    #   ["uniform", "sinusoidal", "logarithmic", "gaussianSwirl", "vortex",
    #    "karmanVortex", "coastalTidal", "linearShear", "doubleGyre",
    #    "gaussianJet", "riverOutflow", "turbulenceNoise"]
    # SIZE_ID: int = 1  # {1: 200x200 m, 2: 2000x2000 m, 3: 20000x20000 m}
    # MASTER_SEED: Optional[int] = None  # set an int for reproducibility

4) **Run batch simulations** (build the benchmark database)

    python run_simulations.py

    # Configure in: code/run_simulations.py
    # -----------------------------------------------------------------------
    # PROBLEM_NAME: str = "zermelo"
    # SCENARIO_TYPE: str = "random"  # 'random' => new scenario per simulation
    # MASTER_SEED: Optional[int] = 1 # global seed; None => random
    # NUM_SIMULATIONS: int = 1000    # per scenario and current type
    #
    # SIZES_ID: Optional[Iterable[int]] = None   # e.g., [1]; None => all sizes
    # CURRENTS_ID: Optional[Iterable[int]] = None  # None => all (0..11)
    #
    # DATABASE_FILE: str = os.path.abspath(os.path.join(
    #     DATA_PATH, "zermelo", "zermelo.db"))
    # DB_RESET: bool = True           # True => clear DB before running
    # PARALLEL_EXECUTION: bool = True # enable parallel execution

5) **Generate reports** (tables/figures for manuscripts)

    python report_simulations.py

---

## Extending Solvers (Zermelo)

1) **Create a solver module**
    
    problems/zermelo/solvers/<your_solver_name>/
      ├── __init__.py
      └── solver.py

2) **Register the solver**
    
    problems/zermelo/problems.py
    # Add to AVAILABLE_SOLVERS with:
    #   - "solve": callable entry point
    #   - "meta":  metadata dict (name, deterministic, params)

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
        # Return control (e.g., heading or heading-rate) at time t, state x.
        return 0.0  # placeholder

    def solve(problem, **kwargs):
        # Prefer using problem.simulate(...) to standardize outputs.
        return problem.simulate(
            sim_id=METADATA["name"],
            state=problem.initial_state,
            max_steps=kwargs.get("max_steps", METADATA["params"]["max_steps"]),
            time_step=kwargs.get("time_step", METADATA["params"]["time_step"]),
            max_execution_time=kwargs.get("max_execution_time", None),
            controller=lambda t, x: control(t, x, problem, **kwargs),
        )

4) **Required outputs** (for fair comparisons)

    # Each solver must yield (directly or via simulate) the following:
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

- Use a fixed `MASTER_SEED` to control pseudo-randomness across scenario generation and (where applicable) solver internals.
- Keep `SCENARIO_TYPE`, `SIZE_ID`, and `CURRENT_TYPE` consistent when comparing methods.
- All solvers are executed under the same protocol (identical scenarios, step sizes, and stopping criteria) to ensure like-for-like, repeatable evaluations.

---
