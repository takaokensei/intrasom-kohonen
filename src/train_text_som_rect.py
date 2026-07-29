"""
Treinamento de variantes RECT (Geometria Retangular) com IntraSOM 1.1.1
para os 4 modelos de texto: 20news_TF-IDF, 20news_SBERT, 6class_TF-IDF, 6class_SBERT.

Gera DUAS variantes por modelo de texto:
  RECT_planar  – lattice='rect', mapshape='planar'
  RECT_toroid  – lattice='rect', mapshape='toroid'

Espelha src/train_text_som_variants.py (HEX_planar) e src/train_som_rect.py (numérico RECT)
usando o mesmo motor IntraSOM 1.1.1 e os mesmos hiperparâmetros que HEX.

Motivo: desde IntraSOM 1.1.1, _rect_dist_tor está corrigida e calcula distâncias toroidais
retangulares corretamente, eliminando o confound MiniSom vs IntraSOM nas comparações.
"""

import os
import sys
import json
import pickle
import shutil
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reproducibility import GLOBAL_SEED, set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
from text_data import load_20news_data, load_6class_data
import intrasom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR      = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR   = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAPSIZE   = (10, 10)
INPUT_LEN = 20

# Variantes a treinar: (sufixo, mapshape)
RECT_VARIANTS = [
    ("RECT_planar", "planar"),
    ("RECT_toroid", "toroid"),
]


def load_embeddings(dataset_name: str, rep_name: str, docs) -> np.ndarray:
    """
    Carrega embeddings 20D usando os transformers .pkl pré-fitados por text_som_clustering.py.
    Não aplica StandardScaler — IntraSOM usa normalization='var' internamente,
    exatamente como HEX_planar e HEX_toroid fazem em train_text_som_variants.py e text_som_clustering.py.
    """
    print(f"  Loading pre-fitted .pkl transformers for {dataset_name} {rep_name}...")

    if rep_name == "TF-IDF":
        vec_path = os.path.join(MAPS_DIR, f"{dataset_name}_tfidf_vectorizer.pkl")
        svd_path = os.path.join(MAPS_DIR, f"{dataset_name}_lsa_svd.pkl")
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(svd_path, "rb") as f:
            svd = pickle.load(f)
        X_tfidf = vectorizer.transform(docs)
        X_emb   = svd.transform(X_tfidf)

    elif rep_name == "SBERT":
        pca_path = os.path.join(MAPS_DIR, f"{dataset_name}_sbert_pca.pkl")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        from sentence_transformers import SentenceTransformer
        print("    Encoding docs with SentenceTransformer (all-MiniLM-L6-v2)...")
        model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
        X_sbert = model_sbert.encode(docs, show_progress_bar=True)
        X_emb   = pca.transform(X_sbert)

    else:
        raise ValueError(f"Unknown representation: {rep_name}")

    return np.asarray(X_emb)


def _move_results(model_name: str) -> None:
    """Move os arquivos do diretório Results/ para outputs/maps/."""
    results_src = os.path.join(os.getcwd(), "Results")
    if not os.path.exists(results_src):
        return
    for fname in os.listdir(results_src):
        if model_name in fname:
            src = os.path.join(results_src, fname)
            dst = os.path.join(MAPS_DIR, fname)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
    # Limpar relatórios temporários
    for fn in os.listdir(os.getcwd()):
        if fn.startswith("Intrasom_report_") and fn.endswith(".txt"):
            try:
                os.remove(fn)
            except Exception:
                pass


def train_text_rect_variant(
    X_emb: np.ndarray,
    labels,
    dataset_name: str,
    rep_name: str,
    variant_key: str,
    mapshape: str,
) -> dict:
    """Treina uma variante IntraSOM RECT para um dataset/representação de texto."""
    model_name = f"SOM_Text_{dataset_name}_{rep_name}_{variant_key}"
    tp = compute_train_params(MAPSIZE)

    print(f"\n  Training IntraSOM {variant_key} for {dataset_name} {rep_name} (10x10, {mapshape})...")

    # DataFrame de embeddings no formato IntraSOM
    df_emb = pd.DataFrame(X_emb)
    df_emb.columns = [f"Dim_{i + 1}" for i in range(df_emb.shape[1])]
    df_emb.index   = [f"Doc_{i + 1}" for i in range(df_emb.shape[0])]

    # Seed fixada antes de build — necessário se SBERT encode() foi chamado antes
    set_global_seed(GLOBAL_SEED)

    som = intrasom.SOMFactory.build(
        data=df_emb,
        mapsize=MAPSIZE,
        mapshape=mapshape,
        lattice="rect",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name=model_name,
        sample_names=list(df_emb.index),
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

    print(f"    {dataset_name}_{rep_name} {variant_key} — QE: {qe:.4f}, TE: {te:.4f}")

    _move_results(model_name)

    return {
        "dataset_name":          dataset_name,
        "representation_name":   rep_name,
        "variant_key":           variant_key,
        "mapsize":               list(MAPSIZE),
        "lattice":               "rect",
        "mapshape":              mapshape,
        "engine":                "intrasom",
        "initialization":        "pca",
        "normalization":         "var",
        "total_epochs":          TOTAL_EPOCHS,
        "train_rough_len":       tp["train_rough_len"],
        "train_finetune_len":    tp["train_finetune_len"],
        "train_rough_radiusin":  tp["train_rough_radiusin"],
        "train_rough_radiusfin": RADIUS_FINAL,
        "quantization_error":    qe,
        "topographic_error":     te,
    }


def main() -> None:
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading 20 Newsgroups data...")
    docs_20news, labels_20news = load_20news_data()

    print("Loading 6-class data...")
    docs_6class, labels_6class = load_6class_data()

    datasets = [
        ("20news", docs_20news, labels_20news),
        ("6class", docs_6class, labels_6class),
    ]
    representations = ["TF-IDF", "SBERT"]

    metrics_list = []

    for dname, docs, labels in datasets:
        for rep_name in representations:
            X_emb = load_embeddings(dname, rep_name, docs)
            for variant_key, mapshape in RECT_VARIANTS:
                metric = train_text_rect_variant(
                    X_emb, labels, dname, rep_name, variant_key, mapshape
                )
                metrics_list.append(metric)

    # Salvar métricas RECT de texto
    metrics_file = os.path.join(METRICS_DIR, "text_rect_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)

    print(f"\nSalvo: {metrics_file}")
    print("\nTreino RECT texto concluído. Parquets em outputs/maps/")
    for m in metrics_list:
        print(
            f"  {m['dataset_name']}_{m['representation_name']} "
            f"{m['variant_key']:12s} — QE={m['quantization_error']:.4f}, TE={m['topographic_error']:.4f}"
        )


if __name__ == "__main__":
    main()
