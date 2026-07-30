"""
umatrix_comparison.py
=====================
Implementa e compara DUAS versões do cálculo da U-matrix reduzida para
malhas retangulares (RECT), conforme identificado no Achado 7 da auditoria:

  (A) umatrix_intrasom  — comportamento atual da biblioteca IntraSOM 1.1.1:
        norma euclidiana bruta dos 8 vizinhos, sem fator 1/sqrt(2)
        nas 4 direções diagonais. Usa build_umatrix() da lib.

  (B) umatrix_classical — reimplementação fiel à Eq. (3) de:
        Costa & Netto (2007), "Segmentação de Mapas Auto-Organizáveis
        com Espaço de Saída 3-D", Rev. Controle & Automação 18(2).
        Divisão por sqrt(2) nas 4 conexões diagonais antes de calcular
        a média com os 4 vizinhos ortogonais.

REPETIÇÃO ESTATÍSTICA (PARTE 2):
  Executa o treinamento dos 12 modelos RECT para 5 SEEDS DIFERENTES:
  seeds = [42, 43, 44, 45, 46] (total de 60 treinamentos).

Métricas por seed e médias/desvios padrão agregados:
  - Correlação de Pearson espacial (r)
  - Diferença relativa média (%): mean(|U_intra - U_class|) / mean(U_class)
  - Razão de escala: mean(U_intra) / mean(U_class)
  - ARI entre segmentações U_A e U_B (K=6)
  - ARI downstream vs rótulos verdadeiros (y): ARI(y, seg_A) vs ARI(y, seg_B)

Saídas:
  outputs/metrics/umatrix_divergence_5seeds.csv  — resumo estatístico (média ± std).
  outputs/metrics/umatrix_divergence_raw_seeds.csv — dados brutos por seed.
  outputs/umatrices/<model>_intrasom.npy          — array 2-D (seed 42).
  outputs/umatrices/<model>_classical.npy         — array 2-D (seed 42).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS

import intrasom

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(WORKSPACE, "outputs", "metrics")
UMAT_DIR   = os.path.join(WORKSPACE, "outputs", "umatrices")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(UMAT_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

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
    # Para cada amostra em X, achar BMU
    normalized_data = som.get_data
    # distânca de cada amostra a todos os neurônios
    dists = np.linalg.norm(normalized_data[:, np.newaxis, :] - codebook_matrix[np.newaxis, :, :], axis=2)
    bmus = np.argmin(dists, axis=1)  # 0-indexed neuron index
    
    umat_seg_flat = umat_seg.flatten()
    sample_labels = umat_seg_flat[bmus]
    return sample_labels


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner across 5 seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed_experiment():
    print("=" * 68)
    print("MULTI-SEED U-MATRIX DIVERGENCE EXPERIMENT (5 SEEDS)")
    print("IntraSOM 1.1.1  vs.  Costa & Netto (2007)")
    print("=" * 68)

    X, y = load_synthetic_control()

    raw_records = []

    total_runs = len(MAP_SIZES) * len(TOPOLOGIES) * len(SEEDS)
    run_count = 0

    for size_name, mapsize in MAP_SIZES.items():
        train_params = compute_train_params(mapsize)
        for top in TOPOLOGIES:
            variant_key = f"RECT_{top}"
            for seed in SEEDS:
                run_count += 1
                model_label = f"SOM_{size_name}_{variant_key}_s{seed}"
                print(f"[{run_count}/{total_runs}] Training {model_label} ...", end=" ", flush=True)

                # Set seed before SOM construction and training
                set_global_seed(seed)

                # Build & Train SOM
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

                # Compute U-matrices
                umat_a = umatrix_intrasom(som)
                umat_b = umatrix_classical(som)

                # Metrics
                flat_a = umat_a.flatten()
                flat_b = umat_b.flatten()

                r_val, _ = pearsonr(flat_a, flat_b)
                mean_diff = np.mean(np.abs(flat_a - flat_b))
                mean_b = np.mean(flat_b)
                rel_diff_pct = (mean_diff / mean_b) * 100 if mean_b > 0 else np.nan
                scale_ratio = np.mean(flat_a) / mean_b if mean_b > 0 else np.nan

                # Segmentations on U-matrix
                seg_a = segment_umatrix(umat_a, k=6, seed=seed)
                seg_b = segment_umatrix(umat_b, k=6, seed=seed)
                seg_ari = adjusted_rand_score(seg_a, seg_b)

                # Downstream clustering quality vs ground truth y
                sample_seg_a = get_sample_clusters_from_umat_seg(som, seg_a, X)
                sample_seg_b = get_sample_clusters_from_umat_seg(som, seg_b, X)

                ari_ground_a = adjusted_rand_score(y, sample_seg_a)
                ari_ground_b = adjusted_rand_score(y, sample_seg_b)

                # Save seed 42 arrays for figure generation compatibility
                if seed == 42:
                    m_name = f"SOM_{size_name}_{variant_key}"
                    np.save(os.path.join(UMAT_DIR, f"{m_name}_intrasom.npy"),  umat_a)
                    np.save(os.path.join(UMAT_DIR, f"{m_name}_classical.npy"), umat_b)
                    np.save(os.path.join(UMAT_DIR, f"{m_name}_diff.npy"),      np.abs(umat_a - umat_b))

                raw_records.append({
                    "size":          size_name,
                    "variant":       variant_key,
                    "topology":      top,
                    "seed":          seed,
                    "pearson_r":     r_val,
                    "rel_diff_pct":  rel_diff_pct,
                    "scale_ratio":   scale_ratio,
                    "seg_ari":       seg_ari,
                    "ari_ground_a":  ari_ground_a,
                    "ari_ground_b":  ari_ground_b,
                })

                print(f"OK (r={r_val:.4f}, rel={rel_diff_pct:.1f}%, scale={scale_ratio:.3f}, seg_ari={seg_ari:.3f})")

    # Convert to DataFrame
    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv(os.path.join(OUT_DIR, "umatrix_divergence_raw_seeds.csv"), index=False)

    # Compute Summary Statistics (mean +- std)
    summary_list = []
    grouped = df_raw.groupby(["size", "variant"])

    for (size_name, var_key), group in grouped:
        summary_list.append({
            "model":              f"SOM_{size_name}_{var_key}",
            "size":               size_name,
            "variant":            var_key,
            "n_seeds":            len(group),
            "pearson_r_mean":     group["pearson_r"].mean(),
            "pearson_r_std":      group["pearson_r"].std(),
            "rel_diff_pct_mean":  group["rel_diff_pct"].mean(),
            "rel_diff_pct_std":   group["rel_diff_pct"].std(),
            "scale_ratio_mean":   group["scale_ratio"].mean(),
            "scale_ratio_std":    group["scale_ratio"].std(),
            "seg_ari_mean":       group["seg_ari"].mean(),
            "seg_ari_std":        group["seg_ari"].std(),
            "ari_ground_a_mean":  group["ari_ground_a"].mean(),
            "ari_ground_a_std":   group["ari_ground_a"].std(),
            "ari_ground_b_mean":  group["ari_ground_b"].mean(),
            "ari_ground_b_std":   group["ari_ground_b"].std(),
            # For backward compatibility with generate_figures.py
            "pearson_r":          group["pearson_r"].mean(),
            "rel_diff_pct":       group["rel_diff_pct"].mean(),
            "scale_ratio":        group["scale_ratio"].mean(),
            "seg_ari":            group["seg_ari"].mean(),
        })

    df_summary = pd.DataFrame(summary_list)
    
    # Custom sort order by size
    size_order = {"5x5": 5, "7x7": 7, "10x10": 10, "12x12": 12, "15x15": 15, "20x20": 20}
    df_summary["dim"] = df_summary["size"].map(size_order)
    df_summary = df_summary.sort_values(by=["dim", "variant"]).drop(columns=["dim"])

    out_csv = os.path.join(OUT_DIR, "umatrix_divergence_5seeds.csv")
    df_summary.to_csv(out_csv, index=False)
    # Also save as umatrix_divergence.csv for generate_figures.py
    df_summary.to_csv(os.path.join(OUT_DIR, "umatrix_divergence.csv"), index=False)

    print("\n" + "=" * 68)
    print(f"[OK] Multi-seed experiment completed successfully!")
    print(f"     Saved summary -> {out_csv}")
    print("=" * 68)

    print("\nSUMMARY STATISTICAL TABLE (Mean +- Std across 5 seeds):")
    cols_show = ["model", "pearson_r_mean", "rel_diff_pct_mean", "scale_ratio_mean", "seg_ari_mean", "ari_ground_a_mean", "ari_ground_b_mean"]
    print(df_summary[cols_show].to_string(index=False))

    return df_summary


if __name__ == "__main__":
    run_multi_seed_experiment()
