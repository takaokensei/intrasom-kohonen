"""
umatrix_comparison.py
=====================
Publication-grade multi-seed experiment and cross-dataset evaluation comparing TWO
versions of the reduced U-matrix for rectangular (RECT) self-organizing maps:

  (A) umatrix_intrasom  — IntraSOM 1.1.1 library method: raw Euclidean distance average
        over 8 neighbors without the 1/sqrt(2) factor for 4 diagonal connections.
  (B) umatrix_classical — Eq. (3) implementation from Costa & Netto (2007):
        division by sqrt(2) on 4 diagonal connections before averaging.

STATISTICAL EXPERIMENT (30 SEEDS, 3 DATASETS):
  Executes SOM training for 12 RECT configurations (6 map sizes x 2 topologies)
  across N=30 seeds (range 42..71) on THREE benchmark datasets:
    1. Synthetic Control (600 samples, 60 attributes, 6 classes)
    2. Wine (178 samples, 13 attributes, 3 classes)
    3. Digits (1797 samples, 64 attributes, 10 classes)

Formal Statistical Features:
  - Paired Wilcoxon signed-rank tests comparing ARI_ground_a vs ARI_ground_b across seeds
  - Benjamini-Hochberg FDR multiple-comparison p-value correction across 12 models
  - 95% Percentile Bootstrap Confidence Intervals (1,000 resamples)
  - Diagnostic CSV tracking unique metric counts and convergence stability
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_wine, load_digits

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params

import intrasom

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(WORKSPACE, "outputs", "metrics")
UMAT_DIR   = os.path.join(WORKSPACE, "outputs", "umatrices")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(UMAT_DIR, exist_ok=True)

SEEDS = list(range(42, 72))  # 30 seeds

MAP_SIZES = {
    "5x5": (5, 5),
    "7x7": (7, 7),
    "10x10": (10, 10),
    "12x12": (12, 12),
    "15x15": (15, 15),
    "20x20": (20, 20)
}

TOPOLOGIES = ["planar", "toroid"]

# ─────────────────────────────────────────────────────────────────────────────
# U-matrix implementations
# ─────────────────────────────────────────────────────────────────────────────

def umatrix_intrasom(som) -> np.ndarray:
    """Versão A: usa o método build_umatrix() da biblioteca IntraSOM 1.1.1."""
    umat_full = som.build_umatrix(expanded=True)
    return np.nanmean(umat_full, axis=2)


def umatrix_classical(som) -> np.ndarray:
    """Versão B: reimplementação fiel à Eq. (3) de Costa & Netto (2007)."""
    codebook = np.asarray(som.codebook.matrix)
    cols_n, rows_n = som.mapsize
    rows, cols = rows_n, cols_n
    weights = codebook.reshape(rows, cols, -1)
    is_toroid = (getattr(som, 'mapshape', 'planar') == 'toroid')

    offsets = [
        ( 0,  1), ( 1,  1), ( 1,  0), ( 1, -1),
        ( 0, -1), (-1, -1), (-1,  0), (-1,  1),
    ]
    SQRT2 = np.sqrt(2.0)
    scale_factors = [1.0, 1.0/SQRT2, 1.0, 1.0/SQRT2,
                     1.0, 1.0/SQRT2, 1.0, 1.0/SQRT2]

    umat = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            dists = []
            for k, (dr, dc) in enumerate(offsets):
                nr, nc = r + dr, c + dc
                if is_toroid:
                    nr %= rows
                    nc %= cols
                elif not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                d = np.linalg.norm(weights[r, c] - weights[nr, nc])
                dists.append(d * scale_factors[k])
            if dists:
                umat[r, c] = np.mean(dists)
    return umat


def segment_umatrix(umat: np.ndarray, k: int = 6, seed: int = 42) -> np.ndarray:
    """Segmenta a U-matrix em k clusters usando K-Means."""
    flat = umat.flatten().reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    return km.fit_predict(flat)


def get_sample_clusters_from_umat_seg(som, umat_seg, X):
    """
    Mapeia os rótulos de cluster de cada neurônio (umat_seg) para as amostras X
    com base no BMU de cada amostra.
    """
    codebook_matrix = np.asarray(som.codebook.matrix)
    cols, rows = som.mapsize
    normalized_data = som.get_data
    dists = np.linalg.norm(normalized_data[:, np.newaxis, :] - codebook_matrix[np.newaxis, :, :], axis=2)
    bmus = np.argmin(dists, axis=1)
    
    umat_seg_flat = umat_seg.flatten()
    sample_labels = umat_seg_flat[bmus]
    return sample_labels


def bootstrap_ci(arr, num_samples=1000, ci=95):
    """Calcula intervalo de confiança percentil bootstrap de 95%."""
    arr = np.asarray(arr)
    if len(arr) < 2 or np.all(arr == arr[0]):
        return arr.mean(), arr.mean()
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(num_samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    lower = np.percentile(boot_means, (100 - ci) / 2.0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return lower, upper


def benjamini_hochberg(p_values):
    """Aplica correção de Benjamini-Hochberg (FDR) sobre p-valores."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    adjusted_p = np.zeros(n)
    
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = sorted_p[i] * n / rank
        cum_min = min(cum_min, adj)
        adjusted_p[sorted_indices[i]] = min(1.0, cum_min)
    return adjusted_p


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_datasets():
    X_syn, y_syn = load_synthetic_control()
    
    wine = load_wine(as_frame=True)
    X_wine, y_wine = wine.data, wine.target
    
    digits = load_digits(as_frame=True)
    X_digits, y_digits = digits.data, digits.target

    return {
        "synthetic_control": (X_syn, y_syn, len(np.unique(y_syn))),
        "wine": (X_wine, y_wine, len(np.unique(y_wine))),
        "digits": (X_digits, y_digits, len(np.unique(y_digits)))
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runner per Dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_for_dataset(ds_name, X, y, n_classes, seeds=SEEDS):
    print("\n" + "=" * 68)
    print(f"RUNNING EXPERIMENT FOR DATASET: {ds_name.upper()} ({len(X)} samples, {X.shape[1]} attrs, {n_classes} classes)")
    print("=" * 68)

    raw_records = []
    total_runs = len(MAP_SIZES) * len(TOPOLOGIES) * len(seeds)
    run_count = 0

    for size_name, mapsize in MAP_SIZES.items():
        train_params = compute_train_params(mapsize)
        for top in TOPOLOGIES:
            variant_key = f"RECT_{top}"
            for seed in seeds:
                run_count += 1
                model_label = f"SOM_{size_name}_{variant_key}_s{seed}"

                set_global_seed(seed)
                som = intrasom.SOMFactory.build(
                    data=X,
                    mapsize=mapsize,
                    mapshape=top,
                    lattice='rect',
                    normalization='var',
                    initialization='random',
                    neighborhood='gaussian',
                    training='batch',
                    name=f"SOM_{size_name}_RECT_{top}",
                    sample_names=list(X.index)
                )
                som.train(previous_epoch=True, **train_params)

                umat_a = umatrix_intrasom(som)
                umat_b = umatrix_classical(som)

                # Save sample numpy arrays for seed 42 (synthetic control)
                if ds_name == "synthetic_control" and seed == 42:
                    np.save(os.path.join(UMAT_DIR, f"SOM_{size_name}_RECT_{top}_intrasom.npy"), umat_a)
                    np.save(os.path.join(UMAT_DIR, f"SOM_{size_name}_RECT_{top}_classical.npy"), umat_b)
                    np.save(os.path.join(UMAT_DIR, f"SOM_{size_name}_RECT_{top}_diff.npy"), np.abs(umat_a - umat_b))

                # Pearson r
                r_val, _ = pearsonr(umat_a.flatten(), umat_b.flatten())

                # Rel diff & scale
                diff_rel_pct = np.mean(np.abs(umat_a - umat_b)) / np.mean(umat_b) * 100.0
                scale_ratio  = np.mean(umat_a) / np.mean(umat_b)

                # Segmentations
                seg_a = segment_umatrix(umat_a, k=n_classes, seed=seed)
                seg_b = segment_umatrix(umat_b, k=n_classes, seed=seed)
                seg_ari = adjusted_rand_score(seg_a.flatten(), seg_b.flatten())

                # Downstream ARI
                samples_a = get_sample_clusters_from_umat_seg(som, seg_a, X)
                samples_b = get_sample_clusters_from_umat_seg(som, seg_b, X)
                ari_gt_a  = adjusted_rand_score(y, samples_a)
                ari_gt_b  = adjusted_rand_score(y, samples_b)

                raw_records.append({
                    "dataset":         ds_name,
                    "model":           f"SOM_{size_name}_{variant_key}",
                    "size":            size_name,
                    "variant":         variant_key,
                    "seed":            seed,
                    "pearson_r":       r_val,
                    "rel_diff_pct":    diff_rel_pct,
                    "scale_ratio":     scale_ratio,
                    "seg_ari":         seg_ari,
                    "ari_ground_a":    ari_gt_a,
                    "ari_ground_b":    ari_gt_b
                })

    df_raw = pd.DataFrame(raw_records)

    # Compute Summary Statistics + Hypothesis Tests + Bootstrap CIs
    summary_list = []
    hyp_list = []
    grouped = df_raw.groupby(["size", "variant"])

    raw_p_values = []
    model_keys = []

    for (size_name, var_key), group in grouped:
        m_key = f"SOM_{size_name}_{var_key}"
        model_keys.append(m_key)
        
        # Paired Wilcoxon test on ari_ground_a vs ari_ground_b
        diffs = group["ari_ground_a"].values - group["ari_ground_b"].values
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            try:
                stat, p_val = wilcoxon(group["ari_ground_a"].values, group["ari_ground_b"].values)
            except Exception:
                p_val = 1.0
        raw_p_values.append(p_val)

    adj_p_values = benjamini_hochberg(raw_p_values)

    p_val_dict = dict(zip(model_keys, adj_p_values))
    raw_p_dict = dict(zip(model_keys, raw_p_values))

    for (size_name, var_key), group in grouped:
        m_key = f"SOM_{size_name}_{var_key}"
        r_ci_l, r_ci_u = bootstrap_ci(group["pearson_r"])
        diff_ci_l, diff_ci_u = bootstrap_ci(group["rel_diff_pct"])
        scale_ci_l, scale_ci_u = bootstrap_ci(group["scale_ratio"])
        seg_ci_l, seg_ci_u = bootstrap_ci(group["seg_ari"])
        gt_a_ci_l, gt_a_ci_u = bootstrap_ci(group["ari_ground_a"])
        gt_b_ci_l, gt_b_ci_u = bootstrap_ci(group["ari_ground_b"])

        p_adj = p_val_dict[m_key]
        p_raw = raw_p_dict[m_key]

        diffs = group["ari_ground_a"].values - group["ari_ground_b"].values
        mean_diff = float(np.mean(diffs))
        mean_abs_diff = float(np.mean(np.abs(diffs)))
        std_diff = float(np.std(diffs, ddof=1))
        cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
        z_stat = float(norm.ppf(1 - p_raw / 2)) if (p_raw > 0 and p_raw < 1) else 0.0
        rosenthal_r = z_stat / np.sqrt(len(diffs))

        summary_list.append({
            "dataset":            ds_name,
            "model":              m_key,
            "size":               size_name,
            "variant":            var_key,
            "n_seeds":            len(group),
            "pearson_r_mean":     group["pearson_r"].mean(),
            "pearson_r_std":      group["pearson_r"].std(),
            "pearson_r_ci95":     f"[{r_ci_l:.4f}, {r_ci_u:.4f}]",
            "rel_diff_pct_mean":  group["rel_diff_pct"].mean(),
            "rel_diff_pct_std":   group["rel_diff_pct"].std(),
            "rel_diff_ci95":      f"[{diff_ci_l:.2f}, {diff_ci_u:.2f}]",
            "scale_ratio_mean":   group["scale_ratio"].mean(),
            "scale_ratio_std":    group["scale_ratio"].std(),
            "scale_ci95":         f"[{scale_ci_l:.4f}, {scale_ci_u:.4f}]",
            "seg_ari_mean":       group["seg_ari"].mean(),
            "seg_ari_std":        group["seg_ari"].std(),
            "seg_ari_ci95":       f"[{seg_ci_l:.3f}, {seg_ci_u:.3f}]",
            "ari_ground_a_mean":  group["ari_ground_a"].mean(),
            "ari_ground_a_std":   group["ari_ground_a"].std(),
            "ari_ground_b_mean":  group["ari_ground_b"].mean(),
            "ari_ground_b_std":   group["ari_ground_b"].std(),
            "ari_diff_mean":      mean_diff,
            "ari_abs_diff_mean":  mean_abs_diff,
            "cohen_d":            cohen_d,
            "rosenthal_r":        rosenthal_r,
            "wilcoxon_p_raw":     p_raw,
            "wilcoxon_p_fdr":     p_adj,
            "h0_rejected_fdr":    p_adj < 0.05,
            # Backward compatibility fields
            "pearson_r":          group["pearson_r"].mean(),
            "rel_diff_pct":       group["rel_diff_pct"].mean(),
            "scale_ratio":        group["scale_ratio"].mean(),
            "seg_ari":            group["seg_ari"].mean(),
        })

    df_summary = pd.DataFrame(summary_list)
    size_order = {"5x5": 5, "7x7": 7, "10x10": 10, "12x12": 12, "15x15": 15, "20x20": 20}
    df_summary["dim"] = df_summary["size"].map(size_order)
    df_summary = df_summary.sort_values(by=["dim", "variant"]).drop(columns=["dim"])

    out_csv = os.path.join(OUT_DIR, f"umatrix_divergence_{ds_name}.csv")
    df_summary.to_csv(out_csv, index=False)

    return df_summary, df_raw


# ─────────────────────────────────────────────────────────────────────────────
# Master Execution Flow
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed_experiment():
    datasets = get_datasets()
    
    all_summaries = {}
    all_raws = []

    for ds_name, (X, y, n_classes) in datasets.items():
        df_sum, df_raw = run_experiment_for_dataset(ds_name, X, y, n_classes, seeds=SEEDS)
        all_summaries[ds_name] = df_sum
        all_raws.append(df_raw)

    df_all_raw = pd.concat(all_raws, ignore_index=True)
    df_all_raw.to_csv(os.path.join(OUT_DIR, "umatrix_divergence_raw_seeds_multidataset.csv"), index=False)

    # Save primary dataset summary as umatrix_divergence_5seeds.csv & umatrix_divergence.csv for compatibility
    syn_summary = all_summaries["synthetic_control"]
    syn_summary.to_csv(os.path.join(OUT_DIR, "umatrix_divergence_5seeds.csv"), index=False)
    syn_summary.to_csv(os.path.join(OUT_DIR, "umatrix_divergence.csv"), index=False)

    # Generate Seed Convergence Diagnostics CSV across all datasets
    diag_list = []
    for (ds_name, size_name, var_key), group in df_all_raw.groupby(["dataset", "size", "variant"]):
        n_uniq_r = group["pearson_r"].nunique()
        n_uniq_ari = group["seg_ari"].nunique()
        r_std = group["pearson_r"].std()
        stable = (r_std < 1e-6) or (n_uniq_r <= 2)
        diag_list.append({
            "dataset": ds_name,
            "model": f"SOM_{size_name}_{var_key}",
            "size": size_name,
            "variant": var_key,
            "n_seeds": len(group),
            "n_unique_pearson_r": n_uniq_r,
            "pearson_r_std": r_std,
            "n_unique_seg_ari": n_uniq_ari,
            "seg_ari_std": group["seg_ari"].std(),
            "convergence_status": "stable_basin" if stable else "variable_basin"
        })
    df_diag = pd.DataFrame(diag_list)
    df_diag.to_csv(os.path.join(OUT_DIR, "seed_convergence_diagnostics.csv"), index=False)

    # Master Hypothesis Testing Summary Table
    hyp_records = []
    for ds_name, df_s in all_summaries.items():
        for _, row in df_s.iterrows():
            hyp_records.append({
                "dataset": ds_name,
                "model": row["model"],
                "ari_gt_a_mean": row["ari_ground_a_mean"],
                "ari_gt_b_mean": row["ari_ground_b_mean"],
                "ari_diff_mean": row["ari_diff_mean"],
                "ari_abs_diff_mean": row["ari_abs_diff_mean"],
                "cohen_d": row["cohen_d"],
                "rosenthal_r": row["rosenthal_r"],
                "wilcoxon_p_raw": row["wilcoxon_p_raw"],
                "wilcoxon_p_fdr": row["wilcoxon_p_fdr"],
                "h0_rejected_fdr": row["h0_rejected_fdr"]
            })
    df_hyp = pd.DataFrame(hyp_records)
    df_hyp.to_csv(os.path.join(OUT_DIR, "hypothesis_testing_summary.csv"), index=False)

    print("\n" + "=" * 68)
    print("MASTER EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Datasets evaluated: {list(datasets.keys())}")
    print(f"Seeds evaluated:    {len(SEEDS)} (seeds {SEEDS[0]}..{SEEDS[-1]})")
    print(f"Total SOM models:   {len(df_all_raw)}")
    print("=" * 68)

    print("\nSYNTHETIC CONTROL STATISTICAL SUMMARY (N=30 seeds, Mean +- Std, FDR p-values):")
    cols_show = ["model", "pearson_r_mean", "rel_diff_pct_mean", "scale_ratio_mean", "seg_ari_mean", "ari_ground_a_mean", "ari_ground_b_mean", "wilcoxon_p_fdr", "h0_rejected_fdr"]
    print(syn_summary[cols_show].to_string(index=False))

    return all_summaries


if __name__ == "__main__":
    run_multi_seed_experiment()
