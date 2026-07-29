"""
Treinamento de variantes RECT (Geometria Retangular) com IntraSOM 1.1.1.

Gera DUAS variantes por tamanho de mapa, usando o mesmo motor e os mesmos
hiperparâmetros que train_som.py usa para HEX:

  RECT_planar  – lattice='rect', mapshape='planar'
  RECT_toroid  – lattice='rect', mapshape='toroid'

Motivo da migração MiniSom → IntraSOM:
  A versão 1.1.1 do IntraSOM corrigiu _rect_dist_tor em codebook.py (que usava
  coordenadas hexagonais para um SOM retangular) e adicionou o guard
  `if self.lattice == "hexa"` em calculate_map_dist. Com esses fixes, o RECT
  toroidal real agora funciona corretamente no IntraSOM, eliminando o confound
  de comparar algoritmos diferentes (MiniSom vs IntraSOM) ao analisar HEX vs RECT.
"""

import os
import sys
import shutil
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import GLOBAL_SEED, set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL

import intrasom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR      = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR   = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAP_SIZES = {
    "5x5":   (5, 5),
    "7x7":   (7, 7),
    "10x10": (10, 10),
    "12x12": (12, 12),
    "15x15": (15, 15),
    "20x20": (20, 20),
}

# Variantes a treinar: (sufixo_arquivo, mapshape)
RECT_VARIANTS = [
    ("RECT_planar", "planar"),
    ("RECT_toroid", "toroid"),
]


def _move_results(size_name: str, variant_key: str) -> None:
    """Move os arquivos gerados pelo IntraSOM para outputs/maps com sufixo de variante."""
    results_src = os.path.join(os.getcwd(), "Results")
    if not os.path.exists(results_src):
        return

    # IntraSOM salva com o `name` passado ao SOMFactory.build()
    model_tag = f"SOM_{size_name}_{variant_key}"
    for fname in os.listdir(results_src):
        if model_tag in fname:
            src = os.path.join(results_src, fname)
            dst = os.path.join(MAPS_DIR, fname)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)


def train_rect_variant(
    X: pd.DataFrame,
    y: pd.Series,
    size_name: str,
    mapsize: tuple,
    variant_key: str,
    mapshape: str,
) -> dict:
    """Treina uma variante RECT (planar ou toroid) com IntraSOM e retorna as métricas."""
    cols, rows = mapsize
    tp = compute_train_params(mapsize)

    print(f"  Training {variant_key} ({cols}x{rows}, mapshape={mapshape})...")
    set_global_seed(GLOBAL_SEED)

    som = intrasom.SOMFactory.build(
        data=X,
        mapsize=mapsize,
        mapshape=mapshape,
        lattice="rect",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name=f"SOM_{size_name}_{variant_key}",
        sample_names=list(X.index),
    )

    som.train(previous_epoch=True, **tp)

    qe = som.calculate_quantization_error
    if callable(qe):
        qe = float(qe())
    else:
        qe = float(qe)

    te = som.topographic_error
    if callable(te):
        te = float(te())
    else:
        te = float(te)

    print(f"    {size_name} {variant_key} — QE: {qe:.4f}, TE: {te:.4f}")

    # Mover parquets para outputs/maps
    _move_results(size_name, variant_key)

    return {
        "size_name":             size_name,
        "variant_key":           variant_key,
        "mapsize":               list(mapsize),
        "lattice":               "rect",
        "mapshape":              mapshape,
        "engine":                "intrasom",
        "initialization":        "pca",
        "total_epochs":          TOTAL_EPOCHS,
        "train_rough_len":       tp["train_rough_len"],
        "train_finetune_len":    tp["train_finetune_len"],
        "train_rough_radiusin":  tp["train_rough_radiusin"],
        "train_rough_radiusfin": RADIUS_FINAL,
        "quantization_error":    qe,
        "topographic_error":     te,
    }


def main() -> None:
    X, y = load_synthetic_control()
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    metrics_list = []

    for size_name, mapsize in MAP_SIZES.items():
        print(f"\n{'='*50}")
        print(f"Training RECT SOM: {size_name} {mapsize}")
        print(f"{'='*50}")

        for variant_key, mapshape in RECT_VARIANTS:
            metric = train_rect_variant(X, y, size_name, mapsize, variant_key, mapshape)
            metrics_list.append(metric)

    # Salvar métricas RECT
    metrics_file = os.path.join(METRICS_DIR, "som_rect_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)

    print(f"\nSalvo: {metrics_file}")
    print("\nTreino RECT concluído. Parquets em outputs/maps/")
    for m in metrics_list:
        print(f"  {m['size_name']} {m['variant_key']:12s} — QE={m['quantization_error']:.4f}, TE={m['topographic_error']:.4f}")


if __name__ == "__main__":
    main()
