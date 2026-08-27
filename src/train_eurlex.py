"""
Treinamento do IntraSOM no EURLEX57K (Legislacao da Uniao Europeia).
Inclui extracao de conceitos EUROVOC, grafos de coocorrencia, pureza topologica
e modularidade de comunidades no mapa de Kohonen.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import intrasom
from intrasom.clustering import ClusterFactory
from src.data_loaders.eurlex import load_eurlex57k
from src.graph.eurovoc_coocurrence import build_eurovoc_coocurrence_graph, analyze_eurovoc_modularity
from src.graph.som_graph import build_som_neuron_graph, compute_som_communities

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


def train_eurlex_som(
    max_samples: int = 5000,
    grid_size: Tuple[int, int] = (10, 10)
):
    print("=" * 70)
    print(f"TREINAMENTO DO INTRASOM NO EURLEX57K (N={max_samples})")
    print("=" * 70)

    set_global_seed()
    df_eur = load_eurlex57k()
    if len(df_eur) > max_samples:
        df_eur = df_eur.sample(n=max_samples, random_state=GLOBAL_SEED).reset_index(drop=True)

    docs = df_eur["text"].tolist()
    labels_multi = df_eur["labels"].tolist()
    num_docs = len(docs)

    print(f"Documentos legislativos carregados: {num_docs}.")

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
        name="SOM_Text_eurlex_TF-IDF",
        sample_names=list(df_data_tfidf.index)
    )
    som_tfidf.train(previous_epoch=True)

    results_tfidf = som_tfidf.results_dataframe
    neurons_tfidf = som_tfidf.neurons_dataframe
    qe_tfidf = get_som_metric(som_tfidf, "calculate_quantization_error")
    te_tfidf = get_som_metric(som_tfidf, "calculate_topographic_error")
    print(f"  [TF-IDF] QE={qe_tfidf:.4f} | TE={te_tfidf:.4f}")

    # Salvar parquets TF-IDF
    neurons_tfidf.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_eurlex_TF-IDF_neurons.parquet"))
    results_tfidf.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_eurlex_TF-IDF_results.parquet"))
    with open(os.path.join(MAPS_DIR, "params_SOM_Text_eurlex_TF-IDF.json"), "w", encoding="utf-8") as f:
        json.dump(get_som_params(som_tfidf), f, indent=2)

    # 2. SBERT (all-MiniLM-L6-v2)
    print("\n--- 2. Sentence-BERT (all-MiniLM-L6-v2) ---")
    sbert_cache = os.path.join(MAPS_DIR, f"eurlex_{max_samples}_sbert_raw.npy")
    if os.path.exists(sbert_cache):
        print(f"  Carregando cache SBERT: {sbert_cache}")
        raw_sbert = np.load(sbert_cache)
    else:
        from sentence_transformers import SentenceTransformer
        sbert_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # Truncar textos para primeiros 512 tokens para rapidez
        docs_trunc = [t[:1500] for t in docs]
        raw_sbert = sbert_m.encode(docs_trunc, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
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
        name="SOM_Text_eurlex_SBERT",
        sample_names=list(df_data_sbert.index)
    )
    som_sbert.train(previous_epoch=True)

    results_sbert = som_sbert.results_dataframe
    neurons_sbert = som_sbert.neurons_dataframe
    qe_sbert = get_som_metric(som_sbert, "calculate_quantization_error")
    te_sbert = get_som_metric(som_sbert, "calculate_topographic_error")
    print(f"  [SBERT] QE={qe_sbert:.4f} | TE={te_sbert:.4f}")

    # Salvar parquets SBERT
    neurons_sbert.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_eurlex_SBERT_neurons.parquet"))
    results_sbert.to_parquet(os.path.join(MAPS_DIR, "SOM_Text_eurlex_SBERT_results.parquet"))
    with open(os.path.join(MAPS_DIR, "params_SOM_Text_eurlex_SBERT.json"), "w", encoding="utf-8") as f:
        json.dump(get_som_params(som_sbert), f, indent=2)

    # 3. Analise de Grafo EUROVOC e Modularity de Comunidades
    print("\n--- 3. Analise de Grafo EUROVOC & Topologia SOM ---")
    g_eurovoc = build_eurovoc_coocurrence_graph(labels_multi, min_coocurrence=5)
    modularity_stats = analyze_eurovoc_modularity(g_eurovoc)
    print(f"  Grafo EUROVOC: {modularity_stats['num_nodes']} conceitos, {modularity_stats['num_edges']} arestas.")
    print(f"  Comunidades detectadas: {modularity_stats['num_communities']} (Modularidade Q={modularity_stats['modularity']:.4f})")

    # 4. Grafo de Prototipos SOM NetworkX
    dim_cols = [f"B_Dim_{i+1}" for i in range(20)]
    codebooks = neurons_sbert[dim_cols].values
    g_som = build_som_neuron_graph(codebooks, grid_size)
    comms_som = compute_som_communities(g_som)
    print(f"  Comunidades no SOM SBERT: {len(comms_som)} macro-clusters detectados.")

    # 5. Salvar relatorio consolidado de metricas
    report = {
        "dataset": "eurlex57k",
        "num_samples": num_docs,
        "metrics": {
            "TF_IDF": {"QE": qe_tfidf, "TE": te_tfidf},
            "SBERT": {"QE": qe_sbert, "TE": te_sbert}
        },
        "eurovoc_graph": modularity_stats,
        "som_communities_count": len(comms_som)
    }
    report_path = os.path.join(METRICS_DIR, "eurlex_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nResultados e metricas salvas em: {report_path}")

if __name__ == "__main__":
    train_eurlex_som(max_samples=5000)
