"""
Treinamento de variantes RECT (Geometria Retangular, Topologia Plana)
utilizando MiniSom como segundo motor complementar.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import GLOBAL_SEED, set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
from minisom import MiniSom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAP_SIZES = {
    "5x5":   (5, 5),
    "7x7":   (7, 7),
    "10x10": (10, 10),
    "12x12": (12, 12),
    "15x15": (15, 15),
    "20x20": (20, 20),
}


def build_neurons_list_minisom(som, X, y, cols, rows):
    """
    Constrói a lista de neurônios para o modelo MiniSom RECT.
    - Coordenadas puras retangulares (cx = col, cy = row) sem offset hex em zig-zag.
    - U-Matrix via som.distance_map() [normalizada (cols, rows)].
    - BMUs via som.winner(sample) -> (w_x, w_y).
    - Codebook via som.get_weights()[c_idx, r_idx, :].tolist().
    """
    distance_map = som.distance_map()  # shape (cols, rows)
    weights = som.get_weights()        # shape (cols, rows, input_len)

    bmu_sample_map = {idx + 1: [] for idx in range(cols * rows)}
    bmu_ids_per_sample = []

    for s_idx, (sample_name, sample_row) in enumerate(X.iterrows()):
        w_x, w_y = som.winner(sample_row.values)
        bmu_id = int(w_y * cols + w_x + 1)
        bmu_sample_map[bmu_id].append(s_idx + 1)  # sample_id 1..600
        bmu_ids_per_sample.append(bmu_id)

    # Determinar distribuição de classes por BMU
    df_samples = pd.DataFrame({"BMU": bmu_ids_per_sample, "Class": y})
    counts = df_samples.groupby(["BMU", "Class"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    dominant_class = counts.idxmax(axis=1) if not counts.empty else pd.Series()
    purity = (counts.max(axis=1) / totals) if not counts.empty else pd.Series()

    neurons_list = []
    for idx in range(cols * rows):
        bmu_idx = idx + 1
        r_idx = idx // cols
        c_idx = idx % cols

        cx = float(c_idx)
        cy = float(r_idx)
        umat_val = float(distance_map[c_idx, r_idx])
        codebook_vec = weights[c_idx, r_idx, :].tolist()

        bmu_totals = int(totals.get(bmu_idx, 0))
        bmu_dominant = str(dominant_class.get(bmu_idx, "Nenhum")) if bmu_totals > 0 else "Vazio"
        bmu_purity = float(purity.get(bmu_idx, 0.0))
        sample_ids = bmu_sample_map[bmu_idx]

        neurons_list.append({
            "id": bmu_idx,
            "x": cx,
            "y": cy,
            "row": int(r_idx),
            "col": int(c_idx),
            "umatrix_value": umat_val,
            "dominant_class": bmu_dominant,
            "purity": bmu_purity,
            "total_samples": bmu_totals,
            "sample_ids": sample_ids,
            "codebook": codebook_vec,
        })

    return neurons_list


def train_rect_som(X, y, size_name, mapsize):
    cols, rows = mapsize
    tp = compute_train_params(mapsize)
    radius_in = float(tp["train_rough_radiusin"])
    rough_len = tp["train_rough_len"]

    print(f"Training MiniSom RECT for {size_name} {mapsize}...")
    set_global_seed(GLOBAL_SEED)

    som = MiniSom(
        x=cols,
        y=rows,
        input_len=X.shape[1],
        topology="rectangular",
        neighborhood_function="gaussian",
        random_seed=GLOBAL_SEED,
    )

    # Inicialização linear via PCA
    som.pca_weights_init(X.values)

    # NOTE: learning_rate stays fixed at its initial value (0.5, MiniSom default) throughout
    # training. Calling train_batch_offline(..., num_iteration=1) once per epoch resets its
    # internal decay clock to t=0 each time, so the decay function always returns the
    # unmodified value. This mirrors the professor's spec, which only defines a schedule for
    # the neighborhood radius (80% -> 1), not for the learning rate.
    for epoch in range(TOTAL_EPOCHS):
        if epoch < rough_len:
            progress = epoch / float(rough_len)
            som._sigma = radius_in - progress * (radius_in - 1.0)
        else:
            som._sigma = 1.0

        # Treino batch síncrono sobre todas as amostras por época
        som.train_batch_offline(X.values, num_iteration=1, verbose=False)

    qe = float(som.quantization_error(X.values))
    te = float(som.topographic_error(X.values))

    print(f"  {size_name} RECT_planar (MiniSom) - QE: {qe:.4f}, TE: {te:.4f}")

    neurons_list = build_neurons_list_minisom(som, X, y, cols, rows)

    return {
        "model_data": {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list,
        },
        "metric_data": {
            "size_name": size_name,
            "variant_key": "RECT_planar",
            "mapsize": list(mapsize),
            "lattice": "rect",
            "mapshape": "planar",
            "engine": "minisom",
            "initialization": "pca",
            "total_epochs": TOTAL_EPOCHS,
            "train_rough_len": tp["train_rough_len"],
            "train_finetune_len": tp["train_finetune_len"],
            "train_rough_radiusin": tp["train_rough_radiusin"],
            "train_rough_radiusfin": RADIUS_FINAL,
            "quantization_error": qe,
            "topographic_error": te,
        }
    }


def main():
    X, y = load_synthetic_control()
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    som_rect_models = {}
    metrics_list = []

    for size_name, mapsize in MAP_SIZES.items():
        res = train_rect_som(X, y, size_name, mapsize)
        som_rect_models[size_name] = res["model_data"]
        metrics_list.append(res["metric_data"])

    # Salvar modelos RECT estruturados diretamente para export_data_for_frontend.py
    rect_models_file = os.path.join(MAPS_DIR, "som_rect_models.json")
    with open(rect_models_file, "w", encoding="utf-8") as f:
        json.dump(som_rect_models, f, ensure_ascii=False)

    metrics_file = os.path.join(METRICS_DIR, "som_rect_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4)

    print(f"\nSaved RECT models JSON to {rect_models_file}")
    print(f"Saved RECT metrics to {metrics_file}")


if __name__ == "__main__":
    main()
