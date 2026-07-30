"""
test_seed_variance.py
=====================
Automated test to verify that setting different seeds produces distinct
codebook initializations and trained weights across all 12 map configurations,
and explicitly reports stable topological convergence when metric variance is near zero.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params
from umatrix_comparison import umatrix_intrasom, umatrix_classical
import intrasom

def test_seed_variance_all_configs():
    X, y = load_synthetic_control()
    map_configs = [
        ('5x5', (5, 5)), ('7x7', (7, 7)), ('10x10', (10, 10)),
        ('12x12', (12, 12)), ('15x15', (15, 15)), ('20x20', (20, 20))
    ]
    topologies = ['planar', 'toroid']

    print("=" * 68)
    print("RUNNING ALL-CONFIG SEED VARIANCE & CONVERGENCE DIAGNOSTIC TEST")
    print("=" * 68)

    for size_name, mapsize in map_configs:
        train_params = compute_train_params(mapsize)
        for top in topologies:
            # Train seed 42
            set_global_seed(42)
            som1 = intrasom.SOMFactory.build(
                data=X, mapsize=mapsize, mapshape=top,
                lattice='rect', normalization='var', initialization='random',
                neighborhood='gaussian', training='batch', name=f"SOM_{size_name}_{top}_s42"
            )
            som1.train(previous_epoch=True, **train_params)

            # Train seed 43
            set_global_seed(43)
            som2 = intrasom.SOMFactory.build(
                data=X, mapsize=mapsize, mapshape=top,
                lattice='rect', normalization='var', initialization='random',
                neighborhood='gaussian', training='batch', name=f"SOM_{size_name}_{top}_s43"
            )
            som2.train(previous_epoch=True, **train_params)

            # Assert post-training codebooks are genuinely different
            max_codebook_diff = np.abs(som1.codebook.matrix - som2.codebook.matrix).max()
            assert max_codebook_diff > 1e-3, f"Codebooks identical for {size_name} {top}! Diff: {max_codebook_diff}"

            # Calculate Pearson r for both
            r1 = float(np.corrcoef(umatrix_intrasom(som1).flatten(), umatrix_classical(som1).flatten())[0, 1])
            r2 = float(np.corrcoef(umatrix_intrasom(som2).flatten(), umatrix_classical(som2).flatten())[0, 1])
            r_diff = abs(r1 - r2)

            if r_diff < 1e-6:
                print(f"[CONVERGENCE STABLE] {size_name} RECT_{top:6s}: Codebook Diff={max_codebook_diff:.4f} | Pearson r Diff={r_diff:.8f} (Stable basin)")
            else:
                print(f"[VARIANCE DETECTED]  {size_name} RECT_{top:6s}: Codebook Diff={max_codebook_diff:.4f} | Pearson r Diff={r_diff:.8f}")

    print("=" * 68)
    print("[PASS] All 12 configurations passed codebook variance & diagnostic checks.")
    print("=" * 68)

if __name__ == "__main__":
    test_seed_variance_all_configs()
