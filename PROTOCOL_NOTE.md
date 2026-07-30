# Coexisting SOM Initialization Protocols

**Date:** 2026-07-30  
**Repository:** `intrasom-kohonen`

---

## 1. Overview of Coexisting Protocols

The codebase intentionally contains two coexisting initialization protocols designed for distinct architectural objectives:

| Pipeline Component | Initialization Protocol | Primary Objective | Rationale |
|---|---|---|---|
| **Dashboard & Training Pipeline**<br>(`train_som.py`, `train_som_rect.py`, `train_som_variants.py`, `export_data_for_frontend.py`) | `initialization='pca'` | **Deterministic UI Rendering** | Recommended by academic advisors to ensure that frontend dashboard users observe identical, stable map orientations and cluster boundaries across application reloads without seed dependency. |
| **Statistical Divergence Experiment**<br>(`src/umatrix_comparison.py`, `tests/test_seed_variance.py`) | `initialization='random'` | **Empirical Seed Variance & Convergence Evaluation** | Used specifically for the multi-seed experiment ($N=60$ runs across 5 seeds) to evaluate topological stability and U-matrix divergence under stochastic weight initializations. |

---

## 2. Protocol Interoperability & Code Integrity

- **No Code Conflict**: Both protocols call the underlying IntraSOM 1.1.1 factory (`SOMFactory.build`).
- **Dashboard Integrity**: The web interface visualizes PCA-initialized maps trained via `train_som_variants.py`, preserving deterministic parity.
- **Paper Reproducibility**: The multi-seed statistical findings in `paper/main.tex` and Table I report the random-initialization protocol, with empirical convergence stability documented in `paper/CONVERGENCE_STABILITY_NOTE.md`.
