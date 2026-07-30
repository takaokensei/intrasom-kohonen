"""
test_seed_variance.py
=====================
Automated test to verify that setting different seeds produces distinct
codebook initializations and trained weights when using initialization='random'.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params
import intrasom

def test_seed_variance():
    X, y = load_synthetic_control()
    train_params = compute_train_params((5, 5))

    # Train seed 42
    set_global_seed(42)
    som1 = intrasom.SOMFactory.build(
        data=X, mapsize=(5, 5), mapshape='planar',
        lattice='rect', normalization='var', initialization='random',
        neighborhood='gaussian', training='batch', name="SOM_test_42"
    )
    som1.train(previous_epoch=True, **train_params)

    # Train seed 43
    set_global_seed(43)
    som2 = intrasom.SOMFactory.build(
        data=X, mapsize=(5, 5), mapshape='planar',
        lattice='rect', normalization='var', initialization='random',
        neighborhood='gaussian', training='batch', name="SOM_test_43"
    )
    som2.train(previous_epoch=True, **train_params)

    max_diff = np.abs(som1.codebook.matrix - som2.codebook.matrix).max()
    print(f"Test seed variance max codebook diff: {max_diff:.6f}")

    assert max_diff > 1e-3, f"Codebooks are identical across seeds 42 and 43! Diff: {max_diff}"
    print("[PASS] Seed variance test passed successfully.")

if __name__ == "__main__":
    test_seed_variance()
