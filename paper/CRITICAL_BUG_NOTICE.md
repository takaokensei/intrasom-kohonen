# Critical Methodological Notice: Deterministic PCA Initialization & Seed Variance

**Date:** 2026-07-30  
**Author:** Cauã Vitor  
**Subject:** Correction of zero-variance artifact in 5-seed repetition table

---

## 1. Problem Statement

In the previous iteration of the multi-seed experiment (`umatrix_comparison.py`),
the SOM models were instantiated with `initialization='pca'`.

As verified by source code inspection of `intrasom/codebook.py::pca_linear_initialization`:
```python
pca = PCA(n_components=2, svd_solver="randomized")
pca.fit(centered_data)
eigvec = pca.components_
```
and batch SOM training (`som.train(bootstrap=False)`):
- PCA initialization is **100% deterministic** given the same input dataset $X$.
- Batch SOM training without bootstrapping (`bootstrap=False`, missing data = False) does **not** draw any random samples during weight updates.

Consequently, setting different random seeds (`seed = 42, 43, 44, 45, 46`) under `initialization='pca'` resulted in training the **exact same codebook matrix 5 times** ($\text{max } |\mathbf{W}_{s1} - \mathbf{W}_{s2}| = 0.0000000000$).

The `std = 0.000` values reported in Table I of the previous manuscript draft were therefore an artifact of deterministic initialization, rather than an empirical demonstration of zero variance.

---

## 2. Corrective Action

To introduce genuine stochasticity across random seeds and evaluate true model variance:

1. **Random Initialization**: Replaced `initialization='pca'` with `initialization='random'` in `src/umatrix_comparison.py`.
2. **Variance Verification**: Confirmed that codebook weights differ across seeds ($\text{max } |\mathbf{W}_{42} - \mathbf{W}_{43}| \approx 2.487$).
3. **Automated Test**: Created `tests/test_seed_variance.py` to assert non-zero codebook difference ($\text{diff} > 10^{-4}$) between different seeds before metric aggregation.
4. **Table & Paper Update**: Re-ran all 60 training runs ($12 \text{ models} \times 5 \text{ seeds}$) with `initialization='random'`, updated Table I with real non-zero standard deviations, and added a methodological note in `paper/main.tex`.
