# ===============================================================================
# Batch runner for Zermelo problem scenarios
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   CLI-oriented script to generate and solve many Zermelo scenarios, optionally
#   in parallel, and persist results to an SQLite database. It prepares paths,
#   seeds, and task lists, then dispatches worker executions that load a problem,
#   configure the scenario, solve it with the available solvers, and store
#   scenarios, currents, simulations, and time-series data in the database.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import sys
import os
import sqlite3
import time
import traceback
import logging
from typing import Any, Callable, Iterable, Optional, Tuple, List

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# =======================================================================
# PREPARE CODE TO LOAD MODULES
# =======================================================================
CODE_PATH: str = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH: str = os.path.dirname(CODE_PATH)
FIGURES_PATH: str = os.path.join(FRAMEWORK_PATH, "figures")
DATA_PATH: str = os.path.join(FRAMEWORK_PATH, "data")
sys.path.insert(0, CODE_PATH)

# =======================================================================
# LOGGING
# =======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =======================================================================
# CONFIGURATION
# =======================================================================
PROBLEM_NAME: str = "zermelo"
SCENARIO_TYPE: str = "random"
MASTER_SEED: int = 1
NUM_SIMULATIONS: int = 1000
SIZES_ID: Optional[Iterable[int]] = None  # e.g. [1]
CURRENTS_ID: Optional[Iterable[int]] = None  # e.g. [0, 5, 7, 8, 11]
DATABASE_FILE: str = os.path.abspath(os.path.join(DATA_PATH, "/zermelo/zermelo.db"))
DB_RESET: bool = True
PARALLEL_EXECUTION: bool = True


# =======================================================================
# DATABASE HELPERS
# =======================================================================
def openDb(db_path: str) -> sqlite3.Connection:
    """
    Open (and initialize pragmas for) an SQLite database connection.

    Parameters
    ----------
    db_path : str
        Absolute path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Open connection configured for WAL mode and reasonable timeouts.

    Raises
    ------
    Exception
        Propagates any error during directory creation or connection setup.
    """
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        con = sqlite3.connect(db_path, timeout=60.0)
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=60000;")
        return con
    except Exception as e:
        print(f"[ERROR] main.openDb failed: {e}")
        raise


# =======================================================================
# WORKER FUNCTION
# =======================================================================
def runOneScenario(args: Tuple[str, str, str, str, int, int, int, int]) -> Tuple[bool, int, int, str, int]:
    """
    Execute a single scenario pipeline in an isolated worker process and
    persist results to the database.

    Parameters
    ----------
    args : tuple
        (code_dir_path, db_path, problem_name, scenario_type, size_id,
         current_field_id, scenario_id, seed)

    Returns
    -------
    tuple
        (ok, scenario_id, current_field_id, info, seed) where:
          - ok : bool
          - scenario_id : int
          - current_field_id : int
          - info : str (elapsed time string on success or error trace on failure)
          - seed : int

    Raises
    ------
    Exception
        Propagates any unexpected failure after logging/packaging error info.
    """
    (code_dir_path, db_path, problem_name, scenario_type, size_id,
     current_field_id, scenario_id, seed) = args

    import sys as _sys, os as _os
    if code_dir_path not in _sys.path:
        _sys.path.insert(0, code_dir_path)

    from problems.problems import loadProblem
    from utils.RandomNumberGenerator import RandomNumberGenerator
    from marine.CurrentField import CurrentField

    t0: float = time.perf_counter()
    con: Optional[sqlite3.Connection] = None
    try:
        rng: RandomNumberGenerator = RandomNumberGenerator(seed=seed)

        current_types = CurrentField.getCurrentTypes()
        if current_field_id < 0 or current_field_id >= len(current_types):
            raise ValueError(f"current_field_id fuera de rango: {current_field_id} (tipos={len(current_types)})")
        current_type_str: str = current_types[current_field_id]

        problem, solvers_configuration, solvers_parameters = loadProblem(
            problem_name=problem_name,
            problem_rng=rng
        )
        problem.configure(
            scenario_parameters={"size_id": size_id, "current_type": current_type_str},
            solvers_parameters=solvers_parameters,
            solvers_configuration=solvers_configuration
        )
        problem.create(type=scenario_type)
        solutions_data = problem.solve()

        con = openDb(db_path)
        cur = con.cursor()

        scen_id: int = problem.scenarioToDb(
            connection=con,
            cursor=cur,
            size_id=size_id,
            current_field_id=int(current_field_id),
            initial_heading=problem.getScenario().getInitialHeading(),
            ship_velocity=problem.getScenario().getShipVelocity(),
            current_velocity=problem.getScenario().getCurrentField().getVelocity(),
            goal=problem.getScenario().getGoal(),
            goal_radius=problem.getScenario().getGoalRadius(),
            r_max=problem.getScenario().getRMax(),
            seed=seed,
        )

        current_db_id: int = problem.currentToDb(con, cur, scen_id, seed)

        for _, solution_data in solutions_data.items():
            sim_data = solution_data["simulation_data"]
            solver_instance = solution_data["solver_instance"]
            solver_id: int = solver_instance.id

            simulation_id: int = problem.simulationToDb(
                connection=con,
                cursor=cur,
                scenario_id=scen_id,
                solver_id=solver_id,
                total_time=sim_data.get("total_time", 0.0),
                total_distance=sim_data.get("total_distance", 0.0),
                num_steps=sim_data.get("num_steps", 0),
                execution_time=sim_data.get("execution_time", 0.0),
                goal_objective=1 if sim_data.get("goal_objective", False) else 0,
                distance_to_goal=sim_data.get("distance_to_goal", 0.0),
                navegation_index=sim_data.get("navegation_index", 0.0),
                solver_seed=int(sim_data.get("solver_seed", 0)),
            )

            problem.simulationsDataToDb(
                connection=con,
                cursor=cur,
                simulation_id=simulation_id,
                simulation_data=sim_data
            )

        con.commit()
        if con is not None:
            con.close()

        elapsed: float = time.perf_counter() - t0
        return True, scenario_id, current_field_id, f"{elapsed:.2f}s", seed

    except Exception as e:
        try:
            if con is not None:
                con.rollback()
                con.close()
        except Exception:
            pass
        tb: str = traceback.format_exc()
        return False, scenario_id, current_field_id, f"{repr(e)}\n{tb}", seed


# ===============================================================================
# MAIN
# ===============================================================================
if __name__ == "__main__":
    from problems.problems import loadProblem
    from utils.RandomNumberGenerator import RandomNumberGenerator
    from marine.CurrentField import CurrentField

    if MASTER_SEED is None:
        master_seed: int = np.random.randint(1, int(1e9))
    else:
        master_seed = MASTER_SEED
    parent_rng: RandomNumberGenerator = RandomNumberGenerator(seed=master_seed)

    if CURRENTS_ID is None:
        CURRENTS_ID = range(len(CurrentField.getCurrentTypes()))

    if SIZES_ID is None:
        SIZES_ID = [1, 2, 3]

    if DB_RESET:
        try:
            problem, _, _ = loadProblem(PROBLEM_NAME, parent_rng)
            con = openDb(DATABASE_FILE)
            cur = con.cursor()
            problem.clearDb(con, cur)
            con.close()
            log.info("Base de datos reseteada.")
        except Exception as e:
            log.error(f"No se pudo resetear la base de datos: {e}")

    tasks: List[Tuple[str, str, str, str, int, int, int, int]] = []
    scenario_id: int = 1
    for size_id in SIZES_ID:
        for current_field_id in CURRENTS_ID:
            for _ in range(NUM_SIMULATIONS):
                seed: int = int(parent_rng.integers(1, int(1e9))[0])
                tasks.append(
                    (
                        CODE_PATH,
                        DATABASE_FILE,
                        PROBLEM_NAME,
                        SCENARIO_TYPE,
                        int(size_id),
                        int(current_field_id),
                        scenario_id,
                        seed,
                    )
                )
                scenario_id += 1

    successes: int = 0
    failures: int = 0
    t0_all: float = time.perf_counter()

    if PARALLEL_EXECUTION:
        max_workers: int = min(os.cpu_count() or 1, 32)
        log.info(f"Ejecutando en paralelo con {max_workers} procesos...")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(runOneScenario, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    ok, sid, cfid, info, seed = fut.result()
                    if ok:
                        successes += 1
                        log.info(f"[OK] scenario={sid} current_field_id={cfid} {info} (seed={seed})")
                    else:
                        failures += 1
                        log.error(f"[ERR] scenario={sid} current_field_id={cfid} err={info} (seed={seed})")
                except Exception as e:
                    failures += 1
                    log.error(f"Una tarea falló catastróficamente: {e}")
    else:
        log.info("Ejecutando de forma secuencial...")
        for t in tasks:
            ok, sid, cfid, info, seed = runOneScenario(t)
            if ok:
                successes += 1
                log.info(f"[OK] scenario={sid} current_field_id={cfid} {info} (seed={seed})")
            else:
                failures += 1
                log.error(f"[ERR] scenario={sid} current_field_id={cfid} err={info} (seed={seed})")

    log.info(f"[DONE] total={len(tasks)} ok={successes} err={failures} elapsed={time.perf_counter()-t0_all:.2f}s")
