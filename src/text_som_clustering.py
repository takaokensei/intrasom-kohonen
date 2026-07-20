import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import json
import shutil
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import intrasom
from intrasom.visualization import PlotFactory
from reproducibility import set_global_seed

# ── Parâmetros de treino recomendados pelo Prof. José Alfredo ──────────────
TOTAL_EPOCHS     = 500   # total de épocas (rough + finetune)
ROUGH_FRACTION   = 0.30  # 30% na fase rough
RADIUS_INIT_FRAC = 0.80  # raio inicial = 80% do maior lado do mapa
RADIUS_FINAL     = 1     # raio final = 1 neurônio


def compute_train_params(mapsize: tuple) -> dict:
    """Parâmetros explícitos de treino para som.train() a partir do tamanho."""
    rough_len    = round(TOTAL_EPOCHS * ROUGH_FRACTION)
    finetune_len = TOTAL_EPOCHS - rough_len
    radius_in    = max(1, round(RADIUS_INIT_FRAC * max(mapsize)))
    return {
        "train_rough_len":          rough_len,
        "train_rough_radiusin":     radius_in,
        "train_rough_radiusfin":    RADIUS_FINAL,
        "train_finetune_len":       finetune_len,
        "train_finetune_radiusin":  RADIUS_FINAL,
        "train_finetune_radiusfin": RADIUS_FINAL,
    }

# Color palette for news categories (combines both datasets)
NEWS_COLORS = {
    # 20 Newsgroups (4 classes)
    "Graphics": "#3182bd",     # Blue
    "Space": "#31a354",        # Green
    "Baseball": "#e6550d",     # Orange
    "Mideast": "#756bb1",      # Purple
    # 6-class dataset (Português)
    "Turismo": "#3182bd",      # Blue
    "Esportes": "#31a354",     # Green
    "Policia": "#e6550d",      # Orange
    "Economia": "#756bb1",     # Purple
    "Politica": "#e7ba52",     # Yellow
    "Variedades": "#d6616b"    # Red
}

def load_20news_data():
    """Loads a subset of 20 Newsgroups with 4 distinct categories (400 docs)."""
    print("Fetching 20 Newsgroups subset...")
    categories = [
        'comp.graphics',
        'sci.space',
        'rec.sport.baseball',
        'talk.politics.mideast'
    ]
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)
    docs = []
    labels = []
    class_map = {
        'comp.graphics': 'Graphics',
        'sci.space': 'Space',
        'rec.sport.baseball': 'Baseball',
        'talk.politics.mideast': 'Mideast'
    }
    counts = {cat: 0 for cat in categories}
    for text, label_idx in zip(newsgroups.data, newsgroups.target):
        cat_name = newsgroups.target_names[label_idx]
        if counts[cat_name] < 100 and len(text.strip()) > 50:
            docs.append(text)
            labels.append(class_map[cat_name])
            counts[cat_name] += 1
    return docs, np.array(labels)

def load_6class_data():
    """Loads the 6-class Brazilian Portuguese news dataset (317 docs, 6 categories)."""
    print("Loading 6-class Excel text dataset...")
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(workspace_dir, "data", "text", "base_dados_textos_6_classes.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset não encontrado em: {file_path}\n"
            "Copie o arquivo 'Base_dados_textos_6_classes.xlsx' para a pasta "
            "'data/text/' na raiz do projeto.\n"
            "Renomeie para: base_dados_textos_6_classes.xlsx (sem espaços/acentos)."
        )
    df = pd.read_excel(file_path)
    
    clean_categories = []
    for cat in df['Categoria']:
        c = str(cat).replace("Polcia", "Policia").replace("Pol\u00edcia", "Policia").replace("Pol\u00e9cia", "Policia")
        c = c.replace("Pol\u00edcia", "Policia")
        c = c.replace("Pol\u00edtica", "Politica").replace("Poltica", "Politica")
        c = c.replace("Polícia e Direitos", "Policia").replace("Política", "Politica")
        c = c.split(" e ")[0]
        
        if "Turismo" in c:
            clean_categories.append("Turismo")
        elif "Esporte" in c:
            clean_categories.append("Esportes")
        elif "Polici" in c or "Policia" in c:
            clean_categories.append("Policia")
        elif "Economia" in c:
            clean_categories.append("Economia")
        elif "Politic" in c or "Politica" in c:
            clean_categories.append("Politica")
        elif "Variedades" in c or "Sociedade" in c:
            clean_categories.append("Variedades")
        else:
            clean_categories.append(c)
            
    df['CleanCategory'] = clean_categories
    docs = df['Texto Expandido'].fillna(df['Texto Original']).fillna('').astype(str).tolist()
    labels = np.array(df['CleanCategory'].tolist())
    return docs, labels

def train_and_plot_text_som(X_embeddings, labels, dataset_name, representation_name, mapsize=(10, 10)):
    """Trains a SOM on text embeddings and plots dominant classes."""
    model_key = f"{dataset_name}_{representation_name}"
    print(f"\nTraining Text SOM for {model_key}...")
    
    # Create DataFrame for IntraSOM input
    df_emb = pd.DataFrame(X_embeddings)
    df_emb.columns = [f"Dim_{i+1}" for i in range(df_emb.shape[1])]
    df_emb.index = [f"Doc_{i+1}" for i in range(df_emb.shape[0])]
    
    # Seed fixada imediatamente antes do build (intrasom usa np.random global)
    set_global_seed()

    train_params = compute_train_params(mapsize)
    print(f"  init: pca | rough: {train_params['train_rough_len']} | "
          f"finetune: {train_params['train_finetune_len']} | "
          f"radius: {train_params['train_rough_radiusin']}->1")

    # Build SOM — initialization='pca' conforme instrução do professor
    som = intrasom.SOMFactory.build(
        data=df_emb,
        mapsize=mapsize,
        mapshape='toroid',
        lattice='hexa',
        normalization='var',
        initialization='pca',     # era 'random' → PCA conforme instrução
        neighborhood='gaussian',
        training='batch',
        name=f"SOM_Text_{model_key}",
        sample_names=list(df_emb.index)
    )

    # Train SOM com parâmetros explícitos de época e raio
    som.train(previous_epoch=True, **train_params)
    
    # Retrieve BMU assignments from results_dataframe
    results_df = som.results_dataframe
    results_df['Class'] = labels
    
    # Calculate dominant class and purity
    counts = results_df.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    dominant_class = counts.idxmax(axis=1)
    purity = counts.max(axis=1) / totals
    
    # Map visual coordinates
    plot_f = PlotFactory(som)
    cols, rows = som.mapsize
    coords = plot_f.generate_hex_lattice(cols, rows)
    
    # Plot dominant class map
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1A1B26')
    ax.set_facecolor('#1A1B26')
    ax.set_aspect('equal')
    
    for idx in range(cols * rows):
        bmu_idx = idx + 1
        cx, cy = coords[idx]
        
        if bmu_idx in totals.index and totals[bmu_idx] > 0:
            cname = dominant_class[bmu_idx]
            face_color = NEWS_COLORS.get(cname, "#aaaaaa")
            pur = purity[bmu_idx]
            num_samples = totals[bmu_idx]
            label_text = f"N{bmu_idx}\n{cname}\n{num_samples}d ({pur*100:.0f}%)"
            alpha = 0.85
            text_color = '#F5F5F5'
        else:
            face_color = "#292E42"
            label_text = f"N{bmu_idx}\nVazio"
            alpha = 0.3
            text_color = '#A9B1D6'
            
        hex_patch = RegularPolygon((cx * 2, cy * 2), numVertices=6, radius=1.05/np.sqrt(3),
                                   facecolor=face_color, edgecolor='#3B4261', alpha=alpha, linewidth=0.5)
        ax.add_patch(hex_patch)
        ax.text(cx * 2, cy * 2, label_text, ha='center', va='center', fontsize=6, fontweight='bold', color=text_color)
        
    ax.set_xlim(coords[:, 0].min() * 2 - 1.5, coords[:, 0].max() * 2 + 1.5)
    ax.set_ylim(coords[:, 1].min() * 2 - 1.5, coords[:, 1].max() * 2 + 1.5)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='#3B4261', label=cname) for cname, color in NEWS_COLORS.items() if cname in labels]
    legend_elements.append(Patch(facecolor='#292E42', edgecolor='#3B4261', label='Vazio', alpha=0.5))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0), title="Categorias", fontsize=10, 
              facecolor='#1F2335', edgecolor='#3B4261', labelcolor='#A9B1D6')
    
    ax.set_title(f"Organização Semântica ({dataset_name} - {representation_name})", fontsize=14, fontweight='bold', pad=20, color='#FFFFFF')
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(workspace_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    save_path = os.path.join(fig_dir, f"som_text_{model_key.lower()}_dominant.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1A1B26')
    plt.close()
    print(f"Dominant class map for {model_key} saved to {save_path}")
    
    # Move params/neurons parquet files to outputs/maps
    results_src_dir = os.path.join(os.getcwd(), "Results")
    if os.path.exists(results_src_dir):
         for file_name in os.listdir(results_src_dir):
             if model_key in file_name or f"Text_{model_key}" in file_name:
                 src_file = os.path.join(results_src_dir, file_name)
                 dest_file = os.path.join(workspace_dir, "outputs", "maps", file_name)
                 if os.path.exists(dest_file):
                     os.remove(dest_file)
                 shutil.move(src_file, dest_file)
                 
    # Move generated plots if any in Plots/
    if os.path.exists("Plots"):
        for root, dirs, files in os.walk("Plots"):
            for file in files:
                if model_key in file:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(fig_dir, file)
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.move(src_file, dest_file)
        shutil.rmtree("Plots")
        
    return results_df, som

def run_experiment_for_dataset(docs, labels, dataset_name):
    """Runs the full TF-IDF and SBERT experiments for a specific dataset."""
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(workspace_dir, "outputs", "maps")
    os.makedirs(models_dir, exist_ok=True)
    import pickle
    
    k = len(np.unique(labels))
    print(f"\n====== Running experiments for {dataset_name} (k={k}) ======")

    # 1. TF-IDF + LSA
    print("\n--- Extracting TF-IDF + LSA ---")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english' if dataset_name == '20news' else None)
    X_tfidf = vectorizer.fit_transform(docs)
    
    svd = TruncatedSVD(n_components=20, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)
    
    # Save vectorizer and SVD
    with open(os.path.join(models_dir, f"{dataset_name}_tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, f"{dataset_name}_lsa_svd.pkl"), "wb") as f:
        pickle.dump(svd, f)
    
    res_tfidf, som_tfidf = train_and_plot_text_som(X_lsa, labels, dataset_name, "TF-IDF")

    # 2. SBERT + PCA
    print("\n--- Extracting Sentence-BERT Embeddings ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_sbert = model.encode(docs, show_progress_bar=True)
    
    pca = PCA(n_components=20, random_state=42)
    X_sbert_reduced = pca.fit_transform(X_sbert)
    
    # Save SBERT PCA
    with open(os.path.join(models_dir, f"{dataset_name}_sbert_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
        
    # Seed fixada novamente: encode() consumiu estado RNG global
    # (ver reproducibility.py — a seed deve ser reposicionada a cada chamada
    #  de SOMFactory e KMeans para garantir bit-a-bit reproducibilidade)
    set_global_seed()
    res_sbert, som_sbert = train_and_plot_text_som(X_sbert_reduced, labels, dataset_name, "SBERT")

    # Metrics
    from intrasom.clustering import ClusterFactory
    
    # Seed antes do KMeans TF-IDF (ClusterFactory usa KMeans sem random_state)
    set_global_seed()
    cf_tfidf = ClusterFactory(som_tfidf)
    clusters_tfidf = cf_tfidf.kmeans(k=k)
    res_c_tfidf = cf_tfidf.results_cluster(clusters_tfidf, save=False)
    ari_tfidf = adjusted_rand_score(labels, res_c_tfidf[f"{k}_clusters"].values)
    nmi_tfidf = normalized_mutual_info_score(labels, res_c_tfidf[f"{k}_clusters"].values)
    
    # Seed antes do KMeans SBERT
    set_global_seed()
    cf_sbert = ClusterFactory(som_sbert)
    clusters_sbert = cf_sbert.kmeans(k=k)
    res_c_sbert = cf_sbert.results_cluster(clusters_sbert, save=False)
    ari_sbert = adjusted_rand_score(labels, res_c_sbert[f"{k}_clusters"].values)
    nmi_sbert = normalized_mutual_info_score(labels, res_c_sbert[f"{k}_clusters"].values)
    
    print(f"\nComparativo {dataset_name}:")
    print(f"TF-IDF SOM: ARI = {ari_tfidf:.4f}, NMI = {nmi_tfidf:.4f}")
    print(f"SBERT SOM: ARI = {ari_sbert:.4f}, NMI = {nmi_sbert:.4f}")
    
    return {
        "TF_IDF": {"ARI": ari_tfidf, "NMI": nmi_tfidf},
        "SBERT": {"ARI": ari_sbert, "NMI": nmi_sbert}
    }

def run_text_experiments():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from sklearn.decomposition import PCA
    
    # Run 1: 20 Newsgroups (4 classes)
    docs_20news, labels_20news = load_20news_data()
    metrics_20news = run_experiment_for_dataset(docs_20news, labels_20news, "20news")
    
    # Run 2: 6 classes (Local Excel)
    docs_6class, labels_6class = load_6class_data()
    metrics_6class = run_experiment_for_dataset(docs_6class, labels_6class, "6class")
    
    # Save combined metrics
    combined_metrics = {
        "20news": metrics_20news,
        "6class": metrics_6class
    }
    with open(os.path.join(workspace_dir, "outputs", "metrics", "text_clustering_comparison.json"), "w") as f:
        json.dump(combined_metrics, f, indent=4)
        
    print("\nText experiments completed for both datasets!")

if __name__ == "__main__":
    run_text_experiments()
