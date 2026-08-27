"""
Treinamento do IntraSOM no 20 Newsgroups Completo (20 classes, ~18k documentos).
Suporta TF-IDF+LSA, SBERT, BGE-M3 e Gemma-300M com avaliacao harmonizada (k=20).
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
from sklearn.preprocessing import Normalizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import intrasom
from src.text_data import load_20news_full
from src.metrics.entropy import calculate_local_neuron_entropy

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def get_or_compute_embeddings(
    docs: List[str],
    rep_name: str
) -> Tuple[np.ndarray, Any]:
    """
    Gera ou carrega embeddings para o dataset completo e reduz para 20D via PCA/SVD.
    """
    cache_path = os.path.join(MAPS_DIR, f"20news_full_{rep_name.lower().replace('-', '')}_emb.npy")
    pca_path = os.path.join(MAPS_DIR, f"20news_full_{rep_name.lower().replace('-', '')}_pca.pkl")

    if rep_name == "TF-IDF":
        vec_path = os.path.join(MAPS_DIR, "20news_full_tfidf_vectorizer.pkl")
        svd_path = os.path.join(MAPS_DIR, "20news_full_lsa_svd.pkl")
        print("  Gerando TF-IDF + TruncatedSVD (20D)...")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=5,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50_000,
        )
        X_tfidf = vectorizer.fit_transform(docs)
        svd = TruncatedSVD(n_components=20, random_state=GLOBAL_SEED)
        X_20d = svd.fit_transform(X_tfidf)
        with open(vec_path, "wb") as f: pickle.dump(vectorizer, f)
        with open(svd_path, "wb") as f: pickle.dump(svd, f)
        return X_20d.astype(np.float32), svd

    # Modelos Densos (SentenceTransformers)
    if rep_name == "SBERT":
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
    elif rep_name == "BGE-M3":
        model_id = "BAAI/bge-m3"
    elif rep_name == "Gemma-300M":
        model_id = "google/embeddinggemma-300m"
    else:
        raise ValueError(f"Modelo desconhecido: {rep_name}")

    if os.path.exists(cache_path):
        print(f"  Carregando embeddings brutos do cache: {cache_path}")
        raw_embs = np.load(cache_path)
    else:
        print(f"  Calculando embeddings brutos com {model_id}...")
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_id)
        if hasattr(st_model, "max_seq_length"):
            st_model.max_seq_length = 512
        raw_embs = st_model.encode(
            docs,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        np.save(cache_path, raw_embs)
        print(f"  Salvo cache em {cache_path} (shape={raw_embs.shape})")

    # PCA para 20D
    print(f"  Ajustando PCA 20D sobre {raw_embs.shape[1]} dimensoes...")
    pca = PCA(n_components=20, random_state=GLOBAL_SEED)
    X_20d = pca.fit_transform(raw_embs)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    return X_20d.astype(np.float32), pca


def train_and_eval_som(
    X_20d: np.ndarray,
    labels: np.ndarray,
    rep_name: str,
    grid_size: Tuple[int, int] = (10, 10),
    mapshape: str = "toroid",
    lattice: str = "hexa"
) -> Dict[str, Any]:
    """
    Treina o IntraSOM e calcula ARI, NMI, QE, TE e entropia local.
    """
    variant_name = "HEX_toroid" if lattice == "hexa" and mapshape == "toroid" else \
                   "HEX_planar" if lattice == "hexa" and mapshape == "planar" else \
                   "RECT_planar" if lattice == "rect" and mapshape == "planar" else "RECT_toroid"

    print(f"  [Treino] 20NG Full | Rep={rep_name} | Grid={grid_size} | Variante={variant_name}")
    num_docs = len(labels)
    df_data = pd.DataFrame(X_20d)
    df_data.columns = [f"Dim_{i+1}" for i in range(20)]
    df_data.index = [f"Doc_{i+1}" for i in range(num_docs)]

    som = intrasom.SOMFactory.build(
        df_data,
        mapsize=grid_size,
        mapshape=mapshape,
        lattice=lattice,
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        seed=GLOBAL_SEED,
        verbose=False
    )
    som.train()

    results_df = som.results
    neurons_df = som.codebook.neurons

    # 1. K-Means superclustering (k=20) sobre BMUs
    cf = intrasom.clustering.ClusterFactory(som)
    k_clusters = len(np.unique(labels))
    neuron_clusters = cf.kmeans(k=k_clusters)
    cols = som.mapsize[0]
    sample_clusters = np.array([
        neuron_clusters[(bmu - 1) // cols, (bmu - 1) % cols]
        for bmu in results_df["BMU"].values
    ])

    ari = float(adjusted_rand_score(labels, sample_clusters))
    nmi = float(normalized_mutual_info_score(labels, sample_clusters))
    qe = float(som.calculate_quantization_error())
    te = float(som.calculate_topographic_error())

    # 2. Entropia local
    entropy_grid, entropy_dict = calculate_local_neuron_entropy(results_df, labels, grid_size)
    mean_entropy = float(np.mean(list(entropy_dict.values())))

    print(f"    -> ARI={ari:.4f} | NMI={nmi:.4f} | QE={qe:.4f} | TE={te:.4f} | H_medio={mean_entropy:.4f}")

    # Salvar parquets
    suffix = f"_{variant_name}" if variant_name != "HEX_toroid" else ""
    neurons_file = os.path.join(MAPS_DIR, f"SOM_Text_20news_full_{rep_name}{suffix}_neurons.parquet")
    results_file = os.path.join(MAPS_DIR, f"SOM_Text_20news_full_{rep_name}{suffix}_results.parquet")
    params_file = os.path.join(MAPS_DIR, f"params_SOM_Text_20news_full_{rep_name}{suffix}.json")

    neurons_df.to_parquet(neurons_file)
    results_df.to_parquet(results_file)
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(som.params, f, indent=2)

    return {
        "ARI": ari,
        "NMI": nmi,
        "QE": qe,
        "TE": te,
        "Mean_Entropy": mean_entropy
    }


def main():
    print("=" * 70)
    print("TREINAMENTO DO INTRASOM NO 20 NEWSGROUPS COMPLETO (20 CLASSES)")
    print("=" * 70)

    docs, labels = load_20news_full(subset="all")
    print(f"Total de documentos: {len(docs)} | Categorias: {len(np.unique(labels))}")

    # 1. TF-IDF baseline
    print("\n--- 1. TF-IDF + LSA ---")
    X_tfidf, _ = get_or_compute_embeddings(docs, "TF-IDF")
    res_tfidf = train_and_eval_som(X_tfidf, labels, "TF-IDF")

    # 2. SBERT
    print("\n--- 2. Sentence-BERT (all-MiniLM-L6-v2) ---")
    X_sbert, _ = get_or_compute_embeddings(docs, "SBERT")
    res_sbert = train_and_eval_som(X_sbert, labels, "SBERT")

    # Resumo
    summary = {
        "20news_full": {
            "TF_IDF": res_tfidf,
            "SBERT": res_sbert
        }
    }
    summary_path = os.path.join(METRICS_DIR, "20news_full_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTreinamento do 20NG Full concluido com sucesso!")
    print(f"Metricas salvas em: {summary_path}")


if __name__ == "__main__":
    main()
