# Engineering Audit Report — IntraSOM Kohonen Maps Analyzer

**Repository:** https://github.com/takaokensei/intrasom-kohonen  
**Audit Period:** 2026-06 to 2026-07  
**Author:** Cauã Vitor (UFRN — Electrical Engineering)  
**Academic Supervisor:** Prof. Dr. José Alfredo Ferreira Costa (UFRN)

This document records six software engineering anomalies identified and corrected
during the development and audit of the *IntraSOM Kohonen Maps Analyzer* dashboard.
These findings are implementation bugs and integration issues, not scientific
contributions. See `paper/main.tex` for the scientific contribution (Finding 7 —
the diagonal normalization divergence in the U-matrix).

---

## Finding 1 — Dual Training Engine: MiniSom (RECT) vs IntraSOM (HEX)

| | |
|---|---|
| **Severity** | Critical |
| **Commit** | `2614693` |
| **File** | `src/train_som_rect.py` |

### Problem
Early project versions used a hybrid engine: hexagonal (HEX) models were trained
with IntraSOM, while rectangular (RECT) models were trained with the MiniSom
library (`minisom.MiniSom`). This meant that QE, TE, and cluster metrics for HEX
and RECT variants were computed by different algorithms, making direct geometric
comparison invalid.

### Evidence
```python
# Before (broken)
from minisom import MiniSom
som = MiniSom(x=mapsize[0], y=mapsize[1], input_len=X.shape[1], ...)
som.train_batch(X_norm, ...)
```

### Resolution
All training scripts refactored to use `intrasom.SOMFactory.build()` exclusively:
```python
# After (fixed)
som = intrasom.SOMFactory.build(
    data=X, mapsize=mapsize, mapshape='planar',
    lattice='rect', normalization='var',
    initialization='pca', training='batch'
)
som.train(previous_epoch=True, **train_params)
```
All 4 SOM variants (HEX/RECT × planar/toroid) now share the same Batch SOM
algorithm and PCA initialization.

---

## Finding 2 — Incorrect Toroidal Distance for RECT in Legacy IntraSOM

| | |
|---|---|
| **Severity** | High |
| **Commit** | `2614693` |
| **File** | `intrasom/codebook.py` (upstream library, v1.1.1) |

### Problem
Versions of IntraSOM prior to 1.1.1 computed toroidal RECT distances using
incorrect boundary metrics that distorted continuity at torus edges.

### Evidence
Inspection of `intrasom/codebook.py` in v1.1.1 shows the corrected modular
difference formula for `_rect_dist_tor`:
```python
delta_r = np.minimum(np.abs(r1 - r2), rows - np.abs(r1 - r2))
delta_c = np.minimum(np.abs(c1 - c2), cols - np.abs(c1 - c2))
dist = np.sqrt(delta_r**2 + delta_c**2)
```

### Resolution
Pinned `intrasom>=1.1.1` in `requirements.txt` and reran the full toroidal
boundary test suite. All RECT_toroid models were retrained with the corrected
upstream version.

---

## Finding 3 — RECT U-Matrix Expanded: 6 vs 8 Neighbors

| | |
|---|---|
| **Severity** | Critical |
| **Commit** | `ad876a3` |
| **File** | `src/export_data_for_frontend.py` |

### Problem
The `build_expanded_umatrix_grid()` function used hexagonal 6-neighbor offsets
for **both** HEX and RECT geometries. For a toroidal RECT 10×10 map:
- Correct (8-connected): $4NM = 400$ unique edges
- Actual (6-neighbor): 300 edges — 100 diagonal edges silently missing

### Evidence
```python
# Before (broken): same hex iterator for both lattices
ii = [[1, 1, 0, -1, 0, 1], [1, 0, -1, -1, -1, 0]]
jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]
for k in range(6): ...  # always 6
```

### Resolution
Added lattice dispatch inside `build_expanded_umatrix_grid()`:
```python
if lattice == 'rect':
    rect_offsets = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    for k in range(8): ...  # 8 neighbors
else:
    # hexagonal 6-neighbor loop (unchanged)
```

---

## Finding 4 — Obsolete Manhattan Monkey-Patch in `src/reproducibility.py`

| | |
|---|---|
| **Severity** | High |
| **Commit** | `2614693` |
| **File** | `src/reproducibility.py` |

### Problem
`src/reproducibility.py` dynamically overwrote `Codebook._rect_dist_plan`
with a Manhattan distance implementation (`|dx| + |dy|`) to work around a bug
in older IntraSOM versions. On import, this silently affected **all** RECT_planar
training, replacing the correct Euclidean metric with Manhattan distance.

### Evidence
```python
# Before (broken)
def _manhattan_rect_dist_plan(self, r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

Codebook._rect_dist_plan = _manhattan_rect_dist_plan  # global side-effect
```

### Resolution
Added version gate to disable the override for IntraSOM ≥ 1.1.1:
```python
import intrasom
_INTRASOM_OK = tuple(int(x) for x in intrasom.__version__.split('.')) >= (1, 1, 1)

if not _INTRASOM_OK:
    Codebook._rect_dist_plan = _manhattan_rect_dist_plan
# else: do nothing — IntraSOM 1.1.1 already has the correct implementation
```
All RECT_planar models were subsequently retrained.

---

## Finding 5 — Effective Dimension Divergence: HEX vs RECT for Odd Sizes

| | |
|---|---|
| **Severity** | Medium |
| **Commit** | `c7e5e21` |
| **File** | `frontend/src/components/SOMParamControls.tsx` |

### Problem
IntraSOM's hexagonal geometry requires an **even** number of rows for toroidal
periodic closure (odd-r stagger scheme). When requesting a 5×5 HEX map, the
library internally adjusts to 5×6 (30 neurons). RECT maps remain strictly 5×5
(25 neurons). The UI displayed nominal "5×5" for both without disclosure.

| Nominal | HEX Effective | RECT Effective |
|---|---|---|
| 5×5 | 5×6 (30 neurons) | 5×5 (25 neurons) |
| 7×7 | 7×8 (56 neurons) | 7×7 (49 neurons) |
| 15×15 | 15×16 (240 neurons) | 15×15 (225 neurons) |
| 10×10, 12×12, 20×20 | Equal to RECT | Equal to HEX |

### Resolution
Added an "Effective Dimension (Engine)" informational label in the UI that
displays the actual `cols × rows` read from the exported JSON, replacing the
nominal label for divergent cases.

---

## Finding 6 — Quantization Error (`QE`) Frozen at Zero After Deserialization

| | |
|---|---|
| **Severity** | Critical |
| **Commit** | `688a02f` |
| **File** | `src/evaluate_clusters.py` |

### Problem
In IntraSOM 1.1.1, `calculate_quantization_error` is a `@property` that returns
`self.QE`. This attribute is initialized to `0.0` in `__init__` and updated only
during the training loop (`train_rough`/`train_finetune`). After loading a saved
model with `SOMFactory.load_som()`, `self.QE` is not recomputed — remaining `0.0`.

### Evidence
```python
# Broken: returns 0.0 for loaded models
qe = som.calculate_quantization_error  # always 0.0 post-deserialization
```

### Resolution
Manual recomputation after loading:
```python
normalized_data = som.get_data
bmu_indices = results_df['BMU'].values.astype(int) - 1
codebook_matrix = np.asarray(som.codebook.matrix)
bmu_vectors = codebook_matrix[bmu_indices]
qe = float(np.mean(np.linalg.norm(normalized_data - bmu_vectors, axis=1)))
```
Corrected QE values range from 3.23 to 4.93, consistent with the training loss curves.

---

## Summary Table

| # | Finding | Severity | Commit | Files Affected |
|---|---------|----------|--------|----------------|
| 1 | Dual engine MiniSom/IntraSOM | Critical | `2614693` | `src/train_som_rect.py` |
| 2 | RECT toroidal distance (upstream) | High | `2614693` | `intrasom/codebook.py` (lib) |
| 3 | RECT U-matrix 6 vs 8 neighbors | Critical | `ad876a3` | `src/export_data_for_frontend.py` |
| 4 | Manhattan monkey-patch | High | `2614693` | `src/reproducibility.py` |
| 5 | HEX/RECT effective dimension | Medium | `c7e5e21` | `frontend/src/components/SOMParamControls.tsx` |
| 6 | QE frozen at 0 after load | Critical | `688a02f` | `src/evaluate_clusters.py` |

---

*For the scientific contribution (Finding 7 — diagonal normalization divergence
in the U-matrix), see `paper/main.tex`.*
