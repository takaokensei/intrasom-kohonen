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

Métricas de divergência computadas por modelo:
  - Correlação de Pearson espacial (r) entre as duas U-matrizes.
  - Diferença relativa média: mean(|U_intra - U_class|) / mean(U_class).
  - RMS da diferença absoluta.
  - Número de clusters resultantes (K-Means com K=6) nas duas versões —
    para verificar se a escolha altera a partição.
  - Adjusted Rand Index entre as duas segmentações.

Saídas:
  outputs/metrics/umatrix_divergence.csv  — tabela de divergência por modelo.
  outputs/umatrices/<model>_intrasom.npy  — array (rows, cols) versão A.
  outputs/umatrices/<model>_classical.npy — array (rows, cols) versão B.
  outputs/umatrices/<model>_diff.npy      — diferença absoluta.
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

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control

import intrasom

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR   = os.path.join(WORKSPACE, "outputs", "maps")
OUT_DIR    = os.path.join(WORKSPACE, "outputs", "metrics")
UMAT_DIR   = os.path.join(WORKSPACE, "outputs", "umatrices")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(UMAT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# U-matrix implementations
# ─────────────────────────────────────────────────────────────────────────────

def umatrix_intrasom(som) -> np.ndarray:
    """
    Versão A: utiliza o método build_umatrix() da biblioteca IntraSOM 1.1.1.
    Retorna array 2-D (rows, cols) com os valores da U-matrix reduzida.
    """
    umat_full = som.build_umatrix(expanded=True)
    # build_umatrix retorna (rows, cols, n_neighbors) com expanded=True
    # A redução (média ao longo dos vizinhos) é: np.nanmean(umat_full, axis=2)
    return np.nanmean(umat_full, axis=2)


def umatrix_classical(som) -> np.ndarray:
    """
    Versão B: reimplementação fiel à Eq. (3) de Costa & Netto (2007).

    Para cada neurônio (r, c):
      U(r,c) = mean(d_ortho) + mean(d_diag / sqrt(2))  [ambas as médias
                juntas, equivalente à média ponderada por comprimento de passo]

    Implementação exata:
      - 4 direções ortogonais: fator de escala = 1.0
      - 4 direções diagonais:  fator de escala = 1/sqrt(2) ≈ 0.7071
      - Aplica-se wrap toroidal se mapshape == 'toroid', senão borda aberta.
      - A média final é sobre todos os vizinhos válidos (nanmean).
    """
    codebook = np.asarray(som.codebook.matrix)  # (rows*cols, n_features)
    cols_n, rows_n = som.mapsize  # IntraSOM: mapsize = (cols, rows)
    rows, cols = rows_n, cols_n
    n_neurons = rows * cols

    weights = codebook.reshape(rows, cols, -1)  # (rows, cols, n_features)

    is_toroid = (getattr(som, 'mapshape', 'planar') == 'toroid')

    # 8 offsets em ordem consistente com IntraSOM 1.1.1 build_umatrix rect:
    # [right, down-right, down, down-left, left, up-left, up, up-right]
    offsets = [
        ( 0,  1), ( 1,  1), ( 1,  0), ( 1, -1),
        ( 0, -1), (-1, -1), (-1,  0), (-1,  1),
    ]
    # Fator de escala: 1/sqrt(2) para diagonais (índices ímpares), 1.0 para ortogonais
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
    """
    Segmenta a U-matrix em k clusters usando K-Means sobre os valores
    da U-matrix achatados (rows*cols,). Retorna labels 1-D de comprimento rows*cols.
    """
    flat = umat.flatten().reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    return km.fit_predict(flat)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

# RECT models only (the divergence only occurs in rect lattice)
RECT_VARIANTS = [
    ("RECT_planar",  "_RECT_planar",  "_RECT_planar"),
    ("RECT_toroid",  "_RECT_toroid",  "_RECT_toroid"),
]

SIZES = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]


def load_rect_som(size_name: str, suffix: str, p_suffix: str, X) -> "intrasom.SOM | None":
    neurons_file = os.path.join(MAPS_DIR, f"SOM_{size_name}{suffix}_neurons.parquet")
    results_file = os.path.join(MAPS_DIR, f"SOM_{size_name}{suffix}_results.parquet")
    params_file  = os.path.join(MAPS_DIR, f"params_SOM_{size_name}{p_suffix}.json")

    if not all(os.path.exists(p) for p in [neurons_file, results_file, params_file]):
        return None, None

    neurons_df = pd.read_parquet(neurons_file)
    results_df = pd.read_parquet(results_file)
    with open(params_file) as fh:
        params = json.load(fh)

    som = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
    return som, results_df


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison():
    print("=" * 64)
    print("U-MATRIX DIVERGENCE EXPERIMENT (Achado 7)")
    print("IntraSOM 1.1.1  vs.  Costa & Netto (2007) formulation")
    print("=" * 64)

    X, y = load_synthetic_control()

    records = []

    for size_name in SIZES:
        for variant_key, suffix, p_suffix in RECT_VARIANTS:
            model_label = f"SOM_{size_name}_{variant_key}"
            print(f"\n>> {model_label}")

            som, results_df = load_rect_som(size_name, suffix, p_suffix, X)
            if som is None:
                print(f"  [SKIP] Files not found for {model_label}")
                continue

            # ── Compute the two U-matrices ────────────────────────────────
            print("  Computing IntraSOM U-matrix …", end=" ", flush=True)
            try:
                umat_a = umatrix_intrasom(som)
                print("OK")
            except Exception as e:
                print(f"FAIL: {e}")
                continue

            print("  Computing Classical U-matrix …", end=" ", flush=True)
            umat_b = umatrix_classical(som)
            print("OK")

            # ── Divergence metrics ────────────────────────────────────────
            flat_a = umat_a.flatten()
            flat_b = umat_b.flatten()

            # Pearson correlation
            r_val, p_val = pearsonr(flat_a, flat_b)

            # Relative mean absolute difference
            mean_diff = np.mean(np.abs(flat_a - flat_b))
            mean_classical = np.mean(flat_b)
            rel_diff = mean_diff / mean_classical if mean_classical > 0 else np.nan

            # RMS of absolute difference
            rms_diff = np.sqrt(np.mean((flat_a - flat_b) ** 2))

            # Scale factor: how much larger are intrasom values vs classical on average
            scale_ratio = np.mean(flat_a) / np.mean(flat_b) if np.mean(flat_b) > 0 else np.nan

            # ── Segmentation comparison ───────────────────────────────────
            labels_a = segment_umatrix(umat_a, k=6)
            labels_b = segment_umatrix(umat_b, k=6)
            seg_ari = adjusted_rand_score(labels_a, labels_b)

            # Save arrays
            np.save(os.path.join(UMAT_DIR, f"{model_label}_intrasom.npy"),  umat_a)
            np.save(os.path.join(UMAT_DIR, f"{model_label}_classical.npy"), umat_b)
            np.save(os.path.join(UMAT_DIR, f"{model_label}_diff.npy"),      np.abs(umat_a - umat_b))

            record = {
                "model":          model_label,
                "size":           size_name,
                "variant":        variant_key,
                "rows":           umat_a.shape[0],
                "cols":           umat_a.shape[1],
                "n_neurons":      umat_a.size,
                "pearson_r":      round(r_val,   6),
                "pearson_p":      round(p_val,   8),
                "mean_intrasom":  round(float(np.mean(flat_a)),  6),
                "mean_classical": round(float(np.mean(flat_b)),  6),
                "scale_ratio":    round(float(scale_ratio),      6),
                "mean_abs_diff":  round(float(mean_diff),        6),
                "rel_diff_pct":   round(float(rel_diff * 100),   4),
                "rms_diff":       round(float(rms_diff),         6),
                "seg_ari":        round(float(seg_ari),          6),
            }
            records.append(record)

            print(f"  Pearson r = {r_val:.4f}   |   rel_diff = {rel_diff*100:.2f}%"
                  f"   |   scale_ratio = {scale_ratio:.4f}   |   seg_ARI = {seg_ari:.4f}")

    if not records:
        print("\n[ERROR] No models found. Check outputs/maps/ directory.")
        return None

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "umatrix_divergence.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved divergence table -> {out_csv}")
    print("\nSummary:")
    print(df[["model","pearson_r","rel_diff_pct","scale_ratio","seg_ari"]].to_string(index=False))

    return df


if __name__ == "__main__":
    run_comparison()
