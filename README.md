# NARUVA: A Modular Benchmarking Framework for Marine Navigation

[![Status](https://img.shields.io/badge/status-alpha-informational)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

> Standardized, extensible experiments for marine navigation under ocean currents, starting from the classical **Zermelo** problem.

## Overview
Navigation for autonomous marine vehicles is challenging due to environmental disturbances such as ocean currents. A natural research starting point is the classical **Zermelo navigation problem**, which seeks the **time‑optimal** trajectory of a vessel moving at **constant speed** within a flow field. Despite extensive study, reproducibility and fair benchmarking have often been hindered by ad‑hoc experimental setups (varying domains, current profiles, discretizations).

**NARUVA** provides an **open‑source, modular, and extensible** framework to run, compare, and extend navigation methods in a consistent way.

---

## Supported Problems

- **Zermelo (baseline)** — Time‑optimal navigation at constant speed within a flow field, with a standardized setup enabling fair, repeatable comparisons of solution methods.  
  → See **[README_ZERMELO.md](./README_ZERMELO.md)** for full details: problem definition, solvers, and usage examples.

- **Planning (Zermelo extensions)** — *Under construction*  
  Future directions include: realistic ship geometries, time‑varying current fields (data‑driven), obstacle representation, adaptive time‑stepping, dynamic ship/wind/wave models with trajectory tracking, and energy‑aware metrics.

---

## Quick Start

1) **Clone**
```bash
git clone https://github.com/picud202301/NARUVA.git
cd NARUVA
```

2) **Environment**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt  # if present
```

3) **Run a demo (if available)**
```bash
# Example: launch a minimal Zermelo example module if your package exposes one.
python -m naruva.problems.zermelo.demo  # adjust or remove if not present
```

> If your runners live under `code/` (as in the current layout), see the examples inside **README_ZERMELO.md**.

---

## Why NARUVA?
- **Reproducible**: fixed scenarios and metrics enable apples‑to‑apples comparisons.
- **Modular**: swap solvers and current fields without changing drivers.
- **Extensible**: add new solvers or variants with minimal boilerplate.

---

## Citing
If you use NARUVA, please cite the associated paper:  
*“Standardizing Navigation Algorithms: A Benchmarking Framework for the Zermelo Problem.”*

---

## License
TBD — choose a license (e.g., MIT or Apache‑2.0) that fits your needs.
