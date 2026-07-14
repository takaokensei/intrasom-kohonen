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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import intrasom
from intrasom.visualization import PlotFactory

# Color palette for 4 news categories
NEWS_COLORS = {
    "Graphics": "#3182bd",     # Blue
    "Space": "#31a354",        # Green
    "Baseball": "#e6550d",     # Orange
    "Mideast": "#756bb1"       # Purple
}

def load_news_data():
    """Loads a subset of 20 Newsgroups with 4 distinct categories."""
    print("Fetching 20 Newsgroups subset...")
    categories = [
        'comp.graphics',
        'sci.space',
        'rec.sport.baseball',
        'talk.politics.mideast'
    ]
    
    # Fetch train subset, shuffle, and limit to 100 samples per class to keep it fast
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)
    
    # Filter to get 100 samples per class
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
            
    print(f"Loaded {len(docs)} documents. Distribution: {pd.Series(labels).value_counts().to_dict()}")
    return docs, np.array(labels)

def train_and_plot_text_som(X_embeddings, labels, representation_name, mapsize=(10, 10)):
    """Trains a SOM on text embeddings and plots dominant classes."""
    print(f"\nTraining Text SOM for {representation_name}...")
    
    # Create DataFrame for IntraSOM input
    df_emb = pd.DataFrame(X_embeddings)
    df_emb.columns = [f"Dim_{i+1}" for i in range(df_emb.shape[1])]
    df_emb.index = [f"Doc_{i+1}" for i in range(df_emb.shape[0])]
    
    # Build SOM
    som = intrasom.SOMFactory.build(
        data=df_emb,
        mapsize=mapsize,
        mapshape='toroid',
        lattice='hexa',
        normalization='var',
        initialization='random',
        neighborhood='gaussian',
        training='batch',
        name=f"SOM_Text_{representation_name}",
        sample_names=list(df_emb.index)
    )
    
    # Train SOM
    som.train(train_len_factor=2, previous_epoch=True)
    
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
    ax.set_aspect('equal')
    
    for idx in range(cols * rows):
        bmu_idx = idx + 1
        cx, cy = coords[idx]
        
        if bmu_idx in totals.index and totals[bmu_idx] > 0:
            cname = dominant_class[bmu_idx]
            face_color = NEWS_COLORS[cname]
            pur = purity[bmu_idx]
            num_samples = totals[bmu_idx]
            label_text = f"N{bmu_idx}\n{cname}\n{num_samples}d ({pur*100:.0f}%)"
            alpha = 0.85
        else:
            face_color = "#f0f0f0"
            label_text = f"N{bmu_idx}\nVazio"
            alpha = 0.3
            
        hex_patch = RegularPolygon((cx * 2, cy * 2), numVertices=6, radius=1.05/np.sqrt(3),
                                   facecolor=face_color, edgecolor='#444444', alpha=alpha, linewidth=0.5)
        ax.add_patch(hex_patch)
        ax.text(cx * 2, cy * 2, label_text, ha='center', va='center', fontsize=6, fontweight='bold', color='#111111')
        
    ax.set_xlim(coords[:, 0].min() * 2 - 1.5, coords[:, 0].max() * 2 + 1.5)
    ax.set_ylim(coords[:, 1].min() * 2 - 1.5, coords[:, 1].max() * 2 + 1.5)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='#333333', label=cname) for cname, color in NEWS_COLORS.items()]
    legend_elements.append(Patch(facecolor='#f0f0f0', edgecolor='#444444', label='Vazio', alpha=0.5))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0), title="Categorias", fontsize=10)
    
    ax.set_title(f"Organização Semântica de Notícias ({representation_name})", fontsize=14, fontweight='bold', pad=20)
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(workspace_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    save_path = os.path.join(fig_dir, f"som_text_{representation_name.lower()}_dominant.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dominant class map for {representation_name} saved to {save_path}")
    
    # Move params/neurons parquet files to outputs/maps
    results_src_dir = os.path.join(os.getcwd(), "Results")
    if os.path.exists(results_src_dir):
        for file_name in os.listdir(results_src_dir):
            if representation_name in file_name or f"Text_{representation_name}" in file_name:
                src_file = os.path.join(results_src_dir, file_name)
                dest_file = os.path.join(workspace_dir, "outputs", "maps", file_name)
                # If destination file exists, remove it first
                if os.path.exists(dest_file):
                    os.remove(dest_file)
                shutil.move(src_file, dest_file)
                
    # Move generated plots if any in Plots/
    if os.path.exists("Plots"):
        for root, dirs, files in os.walk("Plots"):
            for file in files:
                if representation_name in file:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(fig_dir, file)
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.move(src_file, dest_file)
        shutil.rmtree("Plots")
        
    return results_df, som

def run_text_experiments():
    docs, labels = load_news_data()
    
    # 1. TF-IDF + LSA Representation
    print("\n--- Extracting TF-IDF + LSA ---")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(docs)
    
    svd = TruncatedSVD(n_components=20, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)
    print("TF-IDF LSA shape:", X_lsa.shape)
    
    res_tfidf, som_tfidf = train_and_plot_text_som(X_lsa, labels, "TF-IDF")
    
    # 2. Sentence-BERT Representation
    print("\n--- Extracting Sentence-BERT Embeddings ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_sbert = model.encode(docs, show_progress_bar=True)
    
    # Apply PCA to reduce to 20 components to make it comparable
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20, random_state=42)
    X_sbert_reduced = pca.fit_transform(X_sbert)
    print("Sentence-BERT embedding shape:", X_sbert_reduced.shape)
    
    res_sbert, som_sbert = train_and_plot_text_som(X_sbert_reduced, labels, "SBERT")
    
    # Calculate Cluster quality of winner neurons for both
    # We map BMU to 4 clusters (since we have 4 categories)
    from intrasom.clustering import ClusterFactory
    
    cf_tfidf = ClusterFactory(som_tfidf)
    clusters_tfidf = cf_tfidf.kmeans(k=4)
    res_c_tfidf = cf_tfidf.results_cluster(clusters_tfidf, save=False)
    ari_tfidf = adjusted_rand_score(labels, res_c_tfidf["4_clusters"].values)
    nmi_tfidf = normalized_mutual_info_score(labels, res_c_tfidf["4_clusters"].values)
    
    cf_sbert = ClusterFactory(som_sbert)
    clusters_sbert = cf_sbert.kmeans(k=4)
    res_c_sbert = cf_sbert.results_cluster(clusters_sbert, save=False)
    ari_sbert = adjusted_rand_score(labels, res_c_sbert["4_clusters"].values)
    nmi_sbert = normalized_mutual_info_score(labels, res_c_sbert["4_clusters"].values)
    
    print("\n=======================================================")
    print("COMPARATIVO DE CLUSTERIZAÇÃO TEXTUAL:")
    print("=======================================================")
    print(f"TF-IDF + LSA SOM: ARI = {ari_tfidf:.4f}, NMI = {nmi_tfidf:.4f}")
    print(f"Sentence-BERT SOM: ARI = {ari_sbert:.4f}, NMI = {nmi_sbert:.4f}")
    
    # Save text results comparison
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(workspace_dir, "outputs", "metrics", "text_clustering_comparison.json"), "w") as f:
        json.dump({
            "TF_IDF": {"ARI": ari_tfidf, "NMI": nmi_tfidf},
            "SBERT": {"ARI": ari_sbert, "NMI": nmi_sbert}
        }, f, indent=4)
        
if __name__ == "__main__":
    run_text_experiments()
