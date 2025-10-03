# ===============================================================================
# Main Execution Script
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This script serves as the main entry point for running problem simulations.
#   It handles the configuration of the problem type (e.g., Zermelo), scenario
#   parameters, and the master seed for reproducibility. It then loads,
#   configures, creates, solves, and plots the specified problem instance.
# ===============================================================================
from __future__ import annotations

"""
Main execution script to run a navigation problem simulation.

This script initializes a random number generator, loads a specified problem,
configures it with scenario parameters, and runs the full create-solve-plot
workflow. Key execution parameters like problem name, scenario type, and
master seed can be configured in the 'CONFIGURE EXECUTION' section.
"""

# =======================================================================
# PREPARE CODE TO LOAD MODULES
# =======================================================================
import sys
import os
from typing import Optional, Dict, Any, Tuple

# Add project paths to the system path for module loading
CODE_PATH: str = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH: str = os.path.dirname(CODE_PATH)
FIGURES_PATH: str = os.path.join(FRAMEWORK_PATH, "figures")
DATA_PATH: str = os.path.join(FRAMEWORK_PATH, "data")
sys.path.insert(0, CODE_PATH)

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np
from utils.RandomNumberGenerator import RandomNumberGenerator
from problems.problems import loadProblem

# =======================================================================
# CONFIGURE EXECUTION
# =======================================================================
PROBLEM_NAME: str = "zermelo"
SCENARIO_TYPE: str = "random"   # {"fixed", "random"}
CURRENT_TYPE: Optional[str] = None  # If None, a type is sampled from:
# ["uniform","sinusoidal","logarithmic","gaussianSwirl","vortex",
#  "karmanVortex","coastalTidal","linearShear","doubleGyre",
#  "gaussianJet","riverOutflow","turbulenceNoise"]
SIZE_ID: int = 1  # {1: 200x200 m, 2: 2000x2000 m, 3: 20000x20000 m}
MASTER_SEED: Optional[int] = None  # Set an int for reproducible runs

# Interesting seeds for testing:
# 775012903: IPOPT solver may fail.
# 396370562: A* and Analytic solvers may fail.
# 668780092: Only the PSO solver finds a solution.
# =======================================================================
# MAIN SCRIPT
# =======================================================================
if __name__ == "__main__":
    try:
        # ---------------------- Master RNG & seeds ----------------------
        master_seed: int
        if MASTER_SEED is None:
            master_seed = np.random.randint(1, int(1e9))
        else:
            master_seed = MASTER_SEED
        problem_rng: RandomNumberGenerator = RandomNumberGenerator(seed=master_seed)

        # ---------------------- Load Problem ----------------------
        problem: Any
        solvers_configuration: Dict
        solvers_parameters: Dict
        problem, solvers_configuration, solvers_parameters = loadProblem(
            problem_name=PROBLEM_NAME,
            problem_rng=problem_rng
        )
        
        if problem is None:
            print('[MAIN][ERROR] Problem loading failed.')
        else:
            # ---------------------- Configure, Create, Solve and Plot ----------------------
            scenario_params: Dict[str, Any] = {'size_id': SIZE_ID, 'current_type': CURRENT_TYPE}
            
            if problem.configure(scenario_parameters=scenario_params,
                                 solvers_parameters=solvers_parameters,
                                 solvers_configuration=solvers_configuration):
                
                if problem.create(type=SCENARIO_TYPE):
                    problem.print()
                    problem.solve()
                    problem.plot()
                else:
                    print('[MAIN][ERROR] Problem creation failed.')
            else:
                print('[MAIN][ERROR] Problem configuration failed.')

    except Exception as e:
        print(f"[ERROR] Main script execution failed: {e}")
        raise