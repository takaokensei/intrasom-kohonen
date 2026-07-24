"""
Treinamento de variantes RECT (Geometria Retangular, Topologia Plana) via MiniSom
para os 4 modelos de texto: 20news_TF-IDF, 20news_SBERT, 6class_TF-IDF, 6class_SBERT.

Espelha a estrutura de src/train_som_rect.py (padrão já validado no Synthetic Control).

Decisão de normalização (Opção A do v9):
  Aplica StandardScaler().fit_transform() sobre X_embeddings antes do treino MiniSom.
  Isso alinha a escala interna do RECT com a normalização 'var' que o IntraSOM aplica
  internamente no treino HEX, tornando os QEs diretamente comparáveis entre motores.
  O scaler fitado é salvo em {dataset}_{rep}_rect_scaler.pkl para ser reutilizado em
  api.py durante a classificação ao vivo com lattice='RECT'.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reproducibility import GLOBAL_SEED, set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
from text_data import load_20news_data, load_6class_data
from minisom import MiniSom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR      = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR   = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAPSIZE   = (10, 10)
INPUT_LEN = 20


def build_neurons_list_minisom_text(som, X_norm, labels, cols, rows):
    """
    Constrói a lista de neurônios para o modelo MiniSom RECT de texto.
    Adapta build_neurons_list_minisom de train_som_rect.py para:
      - doc_indices (lista 0-indexed) em vez de sample_ids.
      - labels passado como array de strings de categoria.
    """
    distance_map = som.distance_map()   # shape (cols, rows), normalizado [0, 1]
    weights      = som.get_weights()    # shape (cols, rows, INPUT_LEN)

    # Mapear documentos para seus BMUs
    bmu_doc_map = {idx + 1: [] for idx in range(cols * rows)}
    bmu_ids_per_doc = []

    for doc_idx, sample in enumerate(X_norm):
        w_x, w_y = som.winner(sample)
        # bmu_id row-major: row * cols + col + 1
        bmu_id = int(w_y * cols + w_x + 1)
        bmu_doc_map[bmu_id].append(doc_idx)  # 0-indexed
        bmu_ids_per_doc.append(bmu_id)

    # Estatísticas de classe por BMU
    df_docs = pd.DataFrame({"BMU": bmu_ids_per_doc, "Class": labels})
    counts = df_docs.groupby(["BMU", "Class"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    dominant_class = counts.idxmax(axis=1) if not counts.empty else pd.Series()
    purity = (counts.max(axis=1) / totals) if not counts.empty else pd.Series()

    neurons_list = []
    for idx in range(cols * rows):
        bmu_idx = idx + 1
        r_idx   = idx // cols
        c_idx   = idx % cols

        # Coordenadas retangulares puras (sem offset hex zig-zag)
        cx = float(c_idx)
        cy = float(r_idx)

        umat_val    = float(distance_map[c_idx, r_idx])
        codebook_vec = weights[c_idx, r_idx, :].tolist()

        bmu_totals   = int(totals.get(bmu_idx, 0))
        bmu_dominant = str(dominant_class.get(bmu_idx, "Nenhum")) if bmu_totals > 0 else "Vazio"
        bmu_purity   = float(purity.get(bmu_idx, 0.0))
        doc_indices  = bmu_doc_map[bmu_idx]   # lista 0-indexed (igual ao HEX no export)

        neurons_list.append({
            "id":             bmu_idx,
            "x":              cx,
            "y":              cy,
            "row":            int(r_idx),
            "col":            int(c_idx),
            "umatrix_value":  umat_val,
            "dominant_class": bmu_dominant,
            "purity":         bmu_purity,
            "total_samples":  bmu_totals,
            "doc_indices":    doc_indices,
            "codebook":       codebook_vec,
        })

    return neurons_list


def train_text_rect_som(X_norm, labels, dataset_name, rep_name):
    """Treina um modelo MiniSom RECT para um dataset/representação de texto."""
    cols, rows = MAPSIZE
    tp = compute_train_params(MAPSIZE)
    radius_in = float(tp["train_rough_radiusin"])
    rough_len = tp["train_rough_len"]

    model_key = f"{dataset_name}_{rep_name}"
    print(f"\nTraining MiniSom RECT for {model_key} ({cols}x{rows})...")

    # Seed reconfigurada aqui — importante se SentenceTransformer.encode() foi chamado
    # antes, pois ele consome o estado RNG global (documentado em text_som_clustering.py).
    set_global_seed(GLOBAL_SEED)

    som = MiniSom(
        x=cols,
        y=rows,
        input_len=INPUT_LEN,
        topology="rectangular",
        neighborhood_function="gaussian",
        random_seed=GLOBAL_SEED,
    )

    # Inicialização linear via PCA (mesmo padrão do HEX e do train_som_rect.py)
    som.pca_weights_init(X_norm)

    # NOTE: learning_rate permanece fixo em 0.5 (default MiniSom) ao longo do treino.
    # Cada chamada train_batch_offline(..., num_iteration=1) reinicia o clock interno
    # para t=0, neutralizando a decay function internamente — apenas som._sigma é
    # controlado manualmente pelo laço externo.
    for epoch in range(TOTAL_EPOCHS):
        if epoch < rough_len:
            progress = epoch / float(rough_len)
            som._sigma = radius_in - progress * (radius_in - 1.0)
        else:
            som._sigma = 1.0

        # Treino batch síncrono (Kohonen 2013) sobre todos os documentos
        som.train_batch_offline(X_norm, num_iteration=1, verbose=False)

    qe = float(som.quantization_error(X_norm))
    te = float(som.topographic_error(X_norm))

    print(f"  {model_key} RECT_planar (MiniSom) - QE: {qe:.4f}, TE: {te:.4f}")

    neurons_list = build_neurons_list_minisom_text(som, X_norm, labels, cols, rows)

    return {
        "model_data": {
            "cols":    cols,
            "rows":    rows,
            "neurons": neurons_list,
        },
        "metric_data": {
            "dataset_name":       dataset_name,
            "representation_name": rep_name,
            "variant_key":        "RECT_planar",
            "mapsize":            list(MAPSIZE),
            "lattice":            "rect",
            "mapshape":           "planar",
            "engine":             "minisom",
            "initialization":     "pca",
            "normalization":      "standard_scaler_z-score",
            "total_epochs":       TOTAL_EPOCHS,
            "train_rough_len":    tp["train_rough_len"],
            "train_finetune_len": tp["train_finetune_len"],
            "train_rough_radiusin":  tp["train_rough_radiusin"],
            "train_rough_radiusfin": RADIUS_FINAL,
            "quantization_error": qe,
            "topographic_error":  te,
        }
    }


def load_embeddings(dataset_name, rep_name, docs, labels):
    """
    Carrega os transformers .pkl já fitados pelo text_som_clustering.py e aplica
    .transform() para obter embeddings 20D exatamente na mesma base vetorial do HEX.
    Isso garante que o espaço de features RECT e HEX são idênticos antes de qualquer
    normalização adicional — importante para a classificação ao vivo via api.py e pca.ts.
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

    # Normalização Z-score (Opção A, v9): padroniza a escala para paridade com
    # normalization='var' do IntraSOM, tornando QE entre HEX e RECT comparáveis.
    # O scaler fitado é salvo para uso em api.py durante classificação ao vivo RECT.
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_emb)

    # Salvar o scaler para reutilização em api.py
    scaler_path = os.path.join(MAPS_DIR, f"{dataset_name}_{rep_name}_rect_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Saved scaler to {os.path.basename(scaler_path)}")

    return X_norm


def main():
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Carregar documentos brutos (os transformers .pkl já estão fitados,
    # mas precisamos dos docs para .transform() e para mapear BMUs a labels)
    print("Loading 20 Newsgroups data...")
    docs_20news, labels_20news = load_20news_data()

    print("Loading 6-class data...")
    docs_6class, labels_6class = load_6class_data()

    datasets = [
        ("20news", docs_20news, labels_20news),
        ("6class", docs_6class, labels_6class),
    ]
    representations = ["TF-IDF", "SBERT"]

    text_rect_models = {}
    metrics_list = []

    for dname, docs, labels in datasets:
        text_rect_models[dname] = {}
        for rep_name in representations:
            X_norm = load_embeddings(dname, rep_name, docs, labels)
            res = train_text_rect_som(X_norm, labels, dname, rep_name)
            text_rect_models[dname][rep_name] = {"RECT_planar": res["model_data"]}
            metrics_list.append(res["metric_data"])

    # Salvar modelos RECT de texto
    rect_file = os.path.join(MAPS_DIR, "text_rect_models.json")
    with open(rect_file, "w", encoding="utf-8") as f:
        json.dump(text_rect_models, f, ensure_ascii=False)

    # Salvar métricas
    metrics_file = os.path.join(METRICS_DIR, "text_rect_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)

    print(f"\nSaved text RECT models JSON to {rect_file}")
    print(f"Saved text RECT metrics to {metrics_file}")


if __name__ == "__main__":
    main()
