"""
Treinamento de variantes HEX_planar (Geometria Hexagonal, Topologia Plana / Sem Toroide)
via IntraSOM para os 4 modelos de texto: 20news_TF-IDF, 20news_SBERT, 6class_TF-IDF, 6class_SBERT.

Espelha src/train_som_variants.py (variantes numéricas do Synthetic Control).
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

MAPSIZE = (10, 10)


def load_embeddings_no_scaler(dataset_name, rep_name, docs):
    """
    Carrega os embeddings 20D exatamente na mesma base vetorial do HEX_toroid,
    sem aplicar StandardScaler (manter normalization='var' do IntraSOM).
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
        model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
        X_sbert = model_sbert.encode(docs, show_progress_bar=True)
        X_emb   = pca.transform(X_sbert)

    else:
        raise ValueError(f"Unknown representation: {rep_name}")

    return X_emb


def train_text_planar_variant(X_embeddings, labels, dataset_name, rep_name):
    """Treina uma variante IntraSOM HEX_planar para um dataset/representação de texto."""
    model_name = f"SOM_Text_{dataset_name}_{rep_name}_HEX_planar"
    tp = compute_train_params(MAPSIZE)

    print(f"\nTraining IntraSOM HEX_planar for {dataset_name} {rep_name} (10x10)...")

    # DataFrame de embeddings formatado para o IntraSOM
    df_emb = pd.DataFrame(X_embeddings)
    df_emb.columns = [f"Dim_{i+1}" for i in range(df_emb.shape[1])]
    df_emb.index = [f"Doc_{i+1}" for i in range(df_emb.shape[0])]

    set_global_seed(GLOBAL_SEED)

    som = intrasom.SOMFactory.build(
        data=df_emb,
        mapsize=MAPSIZE,
        mapshape="planar",
        lattice="hexa",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name=model_name,
        sample_names=list(df_emb.index)
    )

    som.train(previous_epoch=True, **tp)

    qe = som.calculate_quantization_error
    if callable(qe):
        qe = qe()
    te = som.topographic_error
    if callable(te):
        te = te()

    print(f"  {dataset_name}_{rep_name} HEX_planar (IntraSOM) - QE: {qe:.4f}, TE: {te:.4f}")

    # Mover arquivos de resultados do diretório local Results/ para outputs/maps/
    results_src_dir = os.path.join(os.getcwd(), "Results")
    if os.path.exists(results_src_dir):
        for file_name in os.listdir(results_src_dir):
            if model_name in file_name:
                src_file = os.path.join(results_src_dir, file_name)
                dest_file = os.path.join(MAPS_DIR, file_name)
                if os.path.exists(dest_file):
                    os.remove(dest_file)
                shutil.move(src_file, dest_file)

    # Mover relatórios temporários se gerados
    for fn in os.listdir(os.getcwd()):
        if fn.startswith("Intrasom_report_") and fn.endswith(".txt"):
            try:
                os.remove(fn)
            except Exception:
                pass

    return {
        "dataset_name":        dataset_name,
        "representation_name": rep_name,
        "variant_key":         "HEX_planar",
        "mapsize":             list(MAPSIZE),
        "lattice":             "hexa",
        "mapshape":            "planar",
        "engine":              "intrasom",
        "initialization":      "pca",
        "normalization":       "var",
        "total_epochs":        TOTAL_EPOCHS,
        "train_rough_len":     tp["train_rough_len"],
        "train_finetune_len":  tp["train_finetune_len"],
        "train_rough_radiusin":  tp["train_rough_radiusin"],
        "train_rough_radiusfin": RADIUS_FINAL,
        "quantization_error":  float(qe),
        "topographic_error":   float(te),
    }


def main():
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
            X_emb = load_embeddings_no_scaler(dname, rep_name, docs)
            metric_data = train_text_planar_variant(X_emb, labels, dname, rep_name)
            metrics_list.append(metric_data)

    metrics_file = os.path.join(METRICS_DIR, "text_variants_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)

    print(f"\nSaved text planar variants metrics to {metrics_file}")


if __name__ == "__main__":
    main()
