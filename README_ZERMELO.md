# Zermelo Problem Benchmark

[![Module](https://img.shields.io/badge/module-zermelo-blue)]()
[![Status](https://img.shields.io/badge/status-baseline-success)]()

This module provides a **standardized benchmark** for the classical **Zermelo navigation problem**, enabling consistent, reproducible comparisons across heterogeneous solution methods.

## Problem (brief)
Given a 2D flow field **u(x, y)** and a vessel with constant speed **V**, find a heading control that **minimizes travel time** between a start and a goal while respecting domain bounds (and optional constraints).

> Associated paper: *Standardizing Navigation Algorithms: A Benchmarking Framework for the Zermelo Problem*.

---

## Implemented Solvers
Representative strategies included in the benchmark design:
1. **Analytical** solution for **uniform flows**.
2. **A\*-based search** with **analytical guidance**.
3. **Nonlinear optimization** (e.g., with **IPOPT**).
4. **Particle Swarm Optimization (PSO)**.

This mix of deterministic, continuous, and heuristic methods supports comparison on **travel time**, **path length**, and **solution quality**.

---

## Entry Points & Usage (current layout with `code/`)
If your repository keeps runners under `code/`, use:

### Run a single Zermelo instance
```bash
python code/run_problem.py   --problem zermelo   --solver analytical   --flow uniform   --speed 1.0   --start 0,0   --goal 1,1   --dt 0.1   --out results/single/
```

### Batch simulations / benchmarks
```bash
python code/run_simulations.py   --scenario-dir experiments/zermelo_bench   --solvers analytical a_star ipopt pso   --repeats 10   --out results/bench/
```

### Reporting (tables/plots)
```bash
python code/report_simulations.py   --input results/bench   --export reports/   --format html
```

> Argument names are indicative; run each script with `-h/--help` to see the exact interface in your checkout.

---

## Extending Zermelo Solvers
You can add new solution methods by **creating a solver** and **activating** it:

1. **Create** a new solver module under:
```
problems/zermelo/solvers/
```
2. **Activate/register** the new solver in:
```
problems/zermelo/problems.py
```
3. **Interface guidelines** (suggested):
   - Provide a `solve(problem, **kwargs) -> Solution` entry point.
   - Include metadata (name, deterministic/stochastic, required parameters).
   - Export standard metrics so `run_simulations.py` and `report_simulations.py` can compare results consistently.

---

## Tips
- Keep scenarios and seeds fixed for reproducibility.
- When adding time‑varying currents, document the data source and units.
- For fair comparisons, report both **travel time** and **path length**, and include **gap vs. analytical optimum** when available.
