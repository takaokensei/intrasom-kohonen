"""Fixes all non-determinism sources and upstream library bugs in IntraSOM.

Why this module is needed:
    1. Determinism: The `intrasom` library (installed via pip) uses `np.random.rand(...)`
       internally in `codebook.py` for SOM weight initialization and `sklearn.cluster.KMeans`
       without `random_state` in `clustering.py`. Fixing global seed before builds achieves reproducibility.
    2. Monkey Patch: `intrasom.codebook.Codebook._rect_dist_plan` has a bug in v1.0.4.6
       where a generator expression is passed to `np.array(...)` instead of a list comprehension,
       producing a 0-d object array wrapping a generator instead of distance values.
"""

import random
import warnings
import numpy as np

GLOBAL_SEED: int = 42


def _patch_intrasom_rect_dist_plan() -> None:
    """Monkey-patches intrasom.codebook.Codebook._rect_dist_plan to fix the upstream generator bug."""
    try:
        import intrasom.codebook
        def _rect_dist_plan_fixed(self, node_ind):
            rows, cols = self.mapsize
            coordinates = self.generate_rec_lattice(rows, cols)
            return np.array([abs(coordinates[ind] - coordinates[node_ind]).sum()
                for ind in range(len(coordinates))])
        intrasom.codebook.Codebook._rect_dist_plan = _rect_dist_plan_fixed
    except Exception as exc:
        warnings.warn(
            f"Failed to monkey-patch intrasom.codebook.Codebook._rect_dist_plan: {exc}",
            UserWarning
        )


# Apply monkey patch on import
_patch_intrasom_rect_dist_plan()


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Fix all known sources of randomness used by this pipeline.

    Call this immediately before:
      - `intrasom.SOMFactory.build(...)`
      - `ClusterFactory(...).kmeans(...)`
    """
    random.seed(seed)
    np.random.seed(seed)
