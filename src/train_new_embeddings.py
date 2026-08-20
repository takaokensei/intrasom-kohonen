"""
train_new_embeddings.py
=======================
Treinamento dos modelos IntraSOM com os novos embeddings:
1. BAAI/bge-m3 (1024d -> PCA 20d)
2. google/embeddinggemma-300m (768d -> PCA 20d)

Gera as 4 variantes de mapa 10x10 para cada dataset (20news e 6class):
- HEX_toroid
- HEX_planar
- RECT_toroid
- RECT_planar
"""

import os
import sys
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import intrasom
from reproducibility import GLOBAL_SEED, set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
from text_data import load_20news_data, load_6class_data

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "metrics")
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MAPSIZE = (10, 10)

VARIANTS = [
    ("HEX_toroid", "hexa", "toroid", "SOM_Text_{dataset}_{rep}"),
    ("HEX_planar", "hexa", "planar", "SOM_Text_{dataset}_{rep}_HEX_planar"),
    ("RECT_toroid", "rect", "toroid", "SOM_Text_{dataset}_{rep}_RECT_toroid"),
    ("RECT_planar", "rect", "planar", "SOM_Text_{dataset}_{rep}_RECT_planar"),
]


def move_results_for_model(model_name: str):
    """Move arquivos de resultados gerados na pasta Results/ para outputs/maps/."""
    results_src = os.path.join(os.getcwd(), "Results")
    if os.path.exists(results_src):
        for fname in os.listdir(results_src):
            if model_name in fname:
                src = os.path.join(results_src, fname)
                dst = os.path.join(MAPS_DIR, fname)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
    for fn in os.listdir(os.getcwd()):
        if fn.startswith("Intrasom_report_") and fn.endswith(".txt"):
            try:
                os.remove(fn)
            except Exception:
                pass


def train_single_variant(X_emb, labels, dataset_name, rep_name, var_name, lattice, mapshape, name_template):
    model_name = name_template.format(dataset=dataset_name, rep=rep_name)
    tp = compute_train_params(MAPSIZE)

    df_emb = pd.DataFrame(X_emb)
    df_emb.columns = [f"Dim_{i+1}" for i in range(df_emb.shape[1])]
    df_emb.index = [f"Doc_{i+1}" for i in range(df_emb.shape[0])]

    set_global_seed(GLOBAL_SEED)

    som = intrasom.SOMFactory.build(
        data=df_emb,
        mapsize=MAPSIZE,
        mapshape=mapshape,
        lattice=lattice,
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name=model_name,
        sample_names=list(df_emb.index)
    )

    som.train(previous_epoch=True, **tp)

    # Calcular QE e TE
    qe = som.calculate_quantization_error
    if callable(qe):
        qe = qe()
    te = som.topographic_error
    if callable(te):
        te = te()

    # Calcular ARI e NMI agrupando os neurônios em k classes via K-Means (ClusterFactory),
    # idêntico ao procedimento de benchmark do IntraSOM em src/evaluate_clusters.py e src/text_som_clustering.py
    from intrasom.clustering import ClusterFactory
    k = len(np.unique(labels))
    set_global_seed(GLOBAL_SEED)
    cf = ClusterFactory(som)
    neuron_clusters = cf.kmeans(k=k)  # 2D array (rows, cols) com rótulos de cluster 1..k
    cols = som.mapsize[0]
    results_df = som.results_dataframe
    sample_clusters = np.array([
        neuron_clusters[(bmu - 1) // cols, (bmu - 1) % cols]
        for bmu in results_df["BMU"].values
    ])
    ari = float(adjusted_rand_score(labels, sample_clusters))
    nmi = float(normalized_mutual_info_score(labels, sample_clusters))

    move_results_for_model(model_name)

    print(f"    [{var_name:12s}] (k={k}) ARI: {ari:.4f} | NMI: {nmi:.4f} | QE: {qe:.4f} | TE: {te:.4f}")

    return {
        "dataset_name": dataset_name,
        "representation_name": rep_name,
        "variant": var_name,
        "ARI": ari,
        "NMI": nmi,
        "QE": float(qe),
        "TE": float(te)
    }


def main():
    print("=" * 70)
    print("TREINAMENTO DOS NOVOS EMBEDDINGS INTRA-SOM: BGE-M3 & GEMMA-300M")
    print("=" * 70)

    docs_20news, labels_20news = load_20news_data()
    docs_6class, labels_6class = load_6class_data()

    datasets = [
        ("20news", docs_20news, labels_20news),
        ("6class", docs_6class, labels_6class),
    ]

    models_to_train = [
        ("BGE-M3", "BAAI/bge-m3", None),
        ("Gemma-300M", "google/embeddinggemma-300m", "task: classification | query: ")
    ]

    all_metrics = []

    for rep_name, model_id, prefix in models_to_train:
        print(f"\n>>> Carregando modelo Transformer: {model_id} ({rep_name})...")
        st_model = SentenceTransformer(model_id)
        st_model.max_seq_length = 512

        for dataset_name, docs, labels in datasets:
            print(f"\n--- Processando {dataset_name} com {rep_name} ({len(docs)} docs) ---")
            
            # Formatar com prefixo se aplicável
            if prefix:
                formatted_docs = [f"{prefix}{d}" for d in docs]
            else:
                formatted_docs = docs

            print(f"  Extraindo embeddings densos {st_model.get_sentence_embedding_dimension()}D (max_seq_length=512)...")
            dense_embs = st_model.encode(formatted_docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

            print("  Ajustando PCA 20 componentes...")
            pca = PCA(n_components=20, random_state=GLOBAL_SEED)
            X_20d = pca.fit_transform(dense_embs)

            # Salvar PCA transformer
            pca_filename = f"{dataset_name}_{rep_name.lower().replace('-', '')}_pca.pkl"
            pca_filepath = os.path.join(MAPS_DIR, pca_filename)
            with open(pca_filepath, "wb") as f:
                pickle.dump(pca, f)
            print(f"  Salvo PCA em {pca_filepath}")

            # Treinar as 4 variantes SOM
            for var_name, lattice, mapshape, name_template in VARIANTS:
                m = train_single_variant(
                    X_20d, labels, dataset_name, rep_name, var_name, lattice, mapshape, name_template
                )
                all_metrics.append(m)

    # Salvar métricas consolidadas em JSON
    metrics_summary_path = os.path.join(METRICS_DIR, "new_embeddings_metrics.json")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Atualizar text_clustering_comparison.json com as métricas do HEX_toroid
    comp_file = os.path.join(METRICS_DIR, "text_clustering_comparison.json")
    if os.path.exists(comp_file):
        with open(comp_file, "r", encoding="utf-8") as f:
            comp_metrics = json.load(f)
    else:
        comp_metrics = {}

    for item in all_metrics:
        d_name = item["dataset_name"]
        r_name = item["representation_name"]
        v_name = item["variant"]
        if v_name == "HEX_toroid":
            if d_name not in comp_metrics:
                comp_metrics[d_name] = {}
            comp_metrics[d_name][r_name.replace("-", "_")] = {
                "ARI": item["ARI"],
                "NMI": item["NMI"]
            }
            comp_metrics[d_name][r_name] = {
                "ARI": item["ARI"],
                "NMI": item["NMI"]
            }

    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comp_metrics, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Treinamento concluído com sucesso! Métricas salvas em: {metrics_summary_path} e {comp_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
