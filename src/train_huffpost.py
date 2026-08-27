"""
Treinamento do IntraSOM no HuffPost News Category Dataset.
Inclui amostragem estratificada, reducao de dimensionalidade, treino de SOM,
entropia local e analise de trajetorias temporais e semantic drift (2012–2022).
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import intrasom
from intrasom.clustering import ClusterFactory
from src.data_loaders.huffpost import load_huffpost_data
from src.metrics.entropy import calculate_local_neuron_entropy
from src.visualization.huffpost_temporal import compute_temporal_category_trajectories, compute_semantic_drift_distances

GLOBAL_SEED = 42

def set_global_seed():
    np.random.seed(GLOBAL_SEED)

def get_som_metric(som, attr_name):
    val = getattr(som, attr_name, 0.0)
    return float(val() if callable(val) else val)

def get_som_params(som):
    return {
        "mapsize": list(som.mapsize),
        "mapshape": som.mapshape,
        "lattice": getattr(som.codebook, "lattice", "hexa"),
        "neighborhood": "gaussian",
        "normalization": "var",
        "initialization": "pca",
        "training": "batch",
        "name": getattr(som, "name", "SOM_Model")
    }

MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def train_huffpost_som(
    max_samples: int = 5000,
    grid_size: Tuple[int, int] = (10, 10)
):
    print("=" * 70)
    print(f"TREINAMENTO DO INTRASOM NO HUFFPOST (N={max_samples})")
    print("=" * 70)

    set_global_seed()
    df_huff = load_huffpost_data(max_samples=max_samples, min_category_samples=50, random_state=GLOBAL_SEED)
    docs = df_huff["text"].tolist()
    labels = df_huff["category"].to_numpy()
    num_docs = len(docs)
    k_clusters = len(np.unique(labels))

    print(f"Documentos amostrados: {num_docs} em {k_clusters} categorias ({df_huff['year'].min()}-{df_huff['year'].max()}).")

    # 1. TF-IDF + TruncatedSVD (20D)
    print("\n--- 1. TF-IDF + TruncatedSVD (20D) ---")
    tfidf = TfidfVectorizer(max_features=25000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=20, random_state=GLOBAL_SEED)
    X_tfidf_20d = svd.fit_transform(X_tfidf).astype(np.float32)

    df_data_tfidf = pd.DataFrame(X_tfidf_20d)
    df_data_tfidf.columns = [f"Dim_{i+1}" for i in range(20)]
    df_data_tfidf.index = [f"Doc_{i+1}" for i in range(num_docs)]

    set_global_seed()
    som_tfidf = intrasom.SOMFactory.build(
        data=df_data_tfidf,
        mapsize=grid_size,
        mapshape="toroid",
        lattice="hexa",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name="SOM_Text_huffpost_TF-IDF",
        sample_names=list(df_data_tfidf.index)
    )
    som_tfidf.train(previous_epoch=True)

    results_tfidf = som_tfidf.results_dataframe
    neurons_tfidf = som_tfidf.neurons_dataframe
    cf_tfidf = ClusterFactory(som_tfidf)
    n_clust = cf_tfidf.kmeans(k=k_clusters)
    cols = som_tfidf.mapsize[0]
    sample_clusters_tfidf = np.array([
        n_clust[(bmu - 1) // cols, (bmu - 1) % cols]
        for bmu in results_tfidf["BMU"].values
    ])
    ari_tfidf = float(adjusted_rand_score(labels, sample_clusters_tfidf))
    nmi_tfidf = float(normalized_mutual_info_score(labels, sample_clusters_tfidf))
    qe_tfidf = get_som_metric(som_tfidf, "calculate_quantization_error")
    te_tfidf = get_som_metric(som_tfidf, "calculate_topographic_error")
    _, ent_dict_tfidf = calculate_local_neuron_entropy(results_tfidf, labels, grid_size)
    h_tfidf = float(np.mean(list(ent_dict_tfidf.values())))
    print(f"  [TF-IDF] ARI={ari_tfidf:.4f} | NMI={nmi_tfidf:.4f} | QE={qe_tfidf:.4f} | TE={te_tfidf:.4f} | H_medio={h_tfidf:.4f}")

    # Salvar parquets TF-IDF
    neurons_tfidf.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_huffpost_TF-IDF_neurons.parquet"))
    results_tfidf.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_huffpost_TF-IDF_results.parquet"))
    with open(os.path.join(MAPS_DIR, "params_SOM_Text_huffpost_TF-IDF.json"), "w", encoding="utf-8") as f:
        json.dump(get_som_params(som_tfidf), f, indent=2)

    # 2. SBERT (all-MiniLM-L6-v2)
    print("\n--- 2. Sentence-BERT (all-MiniLM-L6-v2) ---")
    sbert_cache = os.path.join(MAPS_DIR, f"huffpost_{max_samples}_sbert_raw.npy")
    if os.path.exists(sbert_cache):
        print(f"  Carregando cache SBERT: {sbert_cache}")
        raw_sbert = np.load(sbert_cache)
    else:
        from sentence_transformers import SentenceTransformer
        sbert_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        raw_sbert = sbert_m.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        np.save(sbert_cache, raw_sbert)

    pca_sbert = PCA(n_components=20, random_state=GLOBAL_SEED)
    X_sbert_20d = pca_sbert.fit_transform(raw_sbert).astype(np.float32)

    df_data_sbert = pd.DataFrame(X_sbert_20d)
    df_data_sbert.columns = [f"Dim_{i+1}" for i in range(20)]
    df_data_sbert.index = [f"Doc_{i+1}" for i in range(num_docs)]

    set_global_seed()
    som_sbert = intrasom.SOMFactory.build(
        data=df_data_sbert,
        mapsize=grid_size,
        mapshape="toroid",
        lattice="hexa",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name="SOM_Text_huffpost_SBERT",
        sample_names=list(df_data_sbert.index)
    )
    som_sbert.train(previous_epoch=True)

    results_sbert = som_sbert.results_dataframe
    neurons_sbert = som_sbert.neurons_dataframe
    cf_sbert = ClusterFactory(som_sbert)
    n_clust_s = cf_sbert.kmeans(k=k_clusters)
    sample_clusters_sbert = np.array([
        n_clust_s[(bmu - 1) // cols, (bmu - 1) % cols]
        for bmu in results_sbert["BMU"].values
    ])
    ari_sbert = float(adjusted_rand_score(labels, sample_clusters_sbert))
    nmi_sbert = float(normalized_mutual_info_score(labels, sample_clusters_sbert))
    qe_sbert = get_som_metric(som_sbert, "calculate_quantization_error")
    te_sbert = get_som_metric(som_sbert, "calculate_topographic_error")
    _, ent_dict_sbert = calculate_local_neuron_entropy(results_sbert, labels, grid_size)
    h_sbert = float(np.mean(list(ent_dict_sbert.values())))
    print(f"  [SBERT] ARI={ari_sbert:.4f} | NMI={nmi_sbert:.4f} | QE={qe_sbert:.4f} | TE={te_sbert:.4f} | H_medio={h_sbert:.4f}")

    # Salvar parquets SBERT
    neurons_sbert.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_huffpost_SBERT_neurons.parquet"))
    results_sbert.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_huffpost_SBERT_results.parquet"))
    with open(os.path.join(MAPS_DIR, "params_SOM_Text_huffpost_SBERT.json"), "w", encoding="utf-8") as f:
        json.dump(get_som_params(som_sbert), f, indent=2)

    # 3. Trajetorias Temporais e Semantic Drift (2012–2022)
    print("\n--- 3. Analise Temporal & Semantic Drift (2012–2022) ---")
    target_categories = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "TECH", "WELLNESS"]
    trajectories = compute_temporal_category_trajectories(
        df_huff, raw_sbert,
        target_categories=target_categories,
        min_samples_per_year=10
    )
    drift = compute_semantic_drift_distances(trajectories)
    print("  Deslocamento semantico acumulado (Drift Euclidiano no espaco SBERT):")
    for cat, d_val in sorted(drift.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:<15}: {d_val:.4f}")

    # 4. Salvar relatorio consolidado de metricas
    report = {
        "dataset": "huffpost",
        "num_samples": num_docs,
        "num_categories": k_clusters,
        "metrics": {
            "TF_IDF": {"ARI": ari_tfidf, "NMI": nmi_tfidf, "QE": qe_tfidf, "TE": te_tfidf, "Mean_Entropy": h_tfidf},
            "SBERT": {"ARI": ari_sbert, "NMI": nmi_sbert, "QE": qe_sbert, "TE": te_sbert, "Mean_Entropy": h_sbert}
        },
        "semantic_drift": drift
    }
    report_path = os.path.join(METRICS_DIR, "huffpost_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nResultados e metricas salvas em: {report_path}")

if __name__ == "__main__":
    train_huffpost_som(max_samples=5000)
