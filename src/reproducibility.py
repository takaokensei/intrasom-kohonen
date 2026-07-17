"""Fixes all non-determinism sources in the IntraSOM text clustering pipeline.

Why this module is needed:
    The `intrasom` library (installed via pip) uses `np.random.rand(...)` internally
    in `codebook.py` for SOM weight initialization and `sklearn.cluster.KMeans`
    without `random_state` in `clustering.py`. Neither accepts a seed parameter.

    Fixing the NumPy global seed IMMEDIATELY before each call to
    `SOMFactory.build()` and `ClusterFactory.kmeans()` is the only way to
    achieve reproducibility given that we cannot modify the library source.

    IMPORTANT: call `set_global_seed()` again after any operation that may
    consume the RNG state (e.g. `SentenceTransformer.encode()`), right before the
    next SOM or KMeans call — otherwise the sequence diverges.
"""

import random
import numpy as np

GLOBAL_SEED: int = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Fix all known sources of randomness used by this pipeline.

    Call this immediately before:
      - `intrasom.SOMFactory.build(...)`
      - `ClusterFactory(...).kmeans(...)`
    """
    random.seed(seed)
    np.random.seed(seed)
