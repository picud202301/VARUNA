<p align="center">
  <img src="logo_varuna.png" alt="VARUNA / VARUNA Logo" width="220">
</p>

# VARUNA: A Modular Benchmarking Framework for  Navigation Algorithms

[![Status](https://img.shields.io/badge/status-alpha-informational)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

> Standardized, extensible experiments for marine navigation under diferent enviroments, starting from the classical **Zermelo** problem.

## Overview
Autonomous marine navigation spans route planning in dynamic environments, compliance with COLREGs, obstacle avoidance, multi-vehicle coordination, and decision-making under uncertainty. Environmental disturbances—currents, wind, and waves—and the trade-offs among travel time, safety, energy, and robustness highlight the need for **standardized and reproducible** benchmarks.  
**VARUNA** is an open-source, **modular** and **extensible** framework that starts from the Zermelo problem and scales to richer settings. It is explicitly oriented toward reproducible results and easy extensibility: datasets and configurations are standardized for like-for-like comparisons, and a clear solver interface allows optimization- and learning-based methods to be added as plug-ins with minimal effort. All solvers run under a common benchmark protocol, enabling fair, repeatable evaluations.

---

## Supported Problems

- **Zermelo (baseline)** — Time-optimal navigation at constant speed within a flow field, with a standardized setup enabling fair, repeatable comparisons of solution methods.  
  → See **[README_ZERMELO.md](./README_ZERMELO.md)** for detailed problem definition, implemented solvers, and usage examples.

- **Planning (Zermelo extensions)** — *Under construction*  
  Roadmap items include realistic ship geometries, time-varying/data-driven current fields, obstacle representations, adaptive time-stepping, coupled ship/wind/wave dynamics with trajectory tracking, and energy-aware metrics.

---
