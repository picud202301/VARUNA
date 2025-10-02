<p align="center">
  <img src="logo_varuna.png" alt="NARUVA / VARUNA Logo" width="220">
</p>

# NARUVA: A Modular Benchmarking Framework for Marine Navigation

[![Status](https://img.shields.io/badge/status-alpha-informational)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

> Standardized, extensible experiments for marine navigation under ocean currents, starting from the classical **Zermelo** problem.

## Overview
Navigation for autonomous marine vehicles is challenging due to environmental disturbances such as ocean currents. A natural research starting point is the classical **Zermelo navigation problem**, which seeks the **time-optimal** trajectory of a vessel moving at **constant speed** within a flow field. Despite extensive study, reproducibility and fair benchmarking have often been hindered by ad-hoc experimental setups (varying domains, current profiles, discretizations).

**NARUVA** provides an **open-source, modular, and extensible** framework to run, compare, and extend navigation methods in a consistent way.

---

## Supported Problems

- **Zermelo (baseline)** — Time-optimal navigation at constant speed within a flow field, with a standardized setup enabling fair, repeatable comparisons of solution methods.  
  → See **[README_ZERMELO.md](./README_ZERMELO.md)** for full details: problem definition, solvers, and usage examples.

- **Planning (Zermelo extensions)** — *Under construction*  
  Future directions include: realistic ship geometries, time-varying current fields (data-driven), obstacle representation, adaptive time-stepping, dynamic ship/wind/wave models with trajectory tracking, and energy-aware metrics.

---

## Quick Start

1) **Clone**
```bash
git clone https://github.com/picud202301/NARUVA.git
cd NARUVA
