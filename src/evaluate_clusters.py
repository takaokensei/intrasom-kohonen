import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control, CLASS_NAMES

import intrasom
from intrasom.clustering import ClusterFactory

def compute_average_neuron_purity(results_df, y):
    """Calculates the average purity of non-empty neurons in the SOM."""
    temp_df = results_df.copy()
    temp_df['Class'] = y
    
    # Group by BMU and Class
    counts = temp_df.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    
    # Exclude empty neurons (though results_df only lists BMUs that have at least one sample anyway)
    counts = counts[totals > 0]
    totals = totals[totals > 0]
    
    purities = counts.max(axis=1) / totals
    return float(purities.mean())

def evaluate_models():
    print("Starting evaluation of all models...")
    X, y = load_synthetic_control()
    
    # Scale data for baselines
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(workspace_dir, "outputs", "maps")
    metrics_dir = os.path.join(workspace_dir, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    sizes = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]
    som_results = []
    
    for size_name in sizes:
        neurons_file = os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet")
        results_file = os.path.join(maps_dir, f"SOM_{size_name}_results.parquet")
        params_file = os.path.join(maps_dir, f"params_SOM_{size_name}.json")
        
        if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
            print(f"SOM {size_name} files not found. Skipping.")
            continue
            
        print(f"Evaluating SOM {size_name}...")
        
        # Load SOM
        neurons_df = pd.read_parquet(neurons_file)
        results_df = pd.read_parquet(results_file)
        params = json.load(open(params_file))
        
        som = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
        
        # Cluster neurons into 6 groups using K-Means
        cf = ClusterFactory(som)
        neuron_clusters = cf.kmeans(k=6)
        res_df = cf.results_cluster(neuron_clusters, save=False)
        
        # Get sample level cluster assignments
        cluster_col = "6_clusters"
        sample_clusters = res_df[cluster_col].values
        
        # Calculate cluster metrics
        ari = adjusted_rand_score(y, sample_clusters)
        nmi = normalized_mutual_info_score(y, sample_clusters)
        sil = silhouette_score(X_scaled, sample_clusters)
        db = davies_bouldin_score(X_scaled, sample_clusters)
        ch = calinski_harabasz_score(X_scaled, sample_clusters)
        
        # Average neuron purity (non-clustered, just winner neurons)
        purity = compute_average_neuron_purity(res_df, y)
        
        # Error metrics (already calculated during training, let's reload them)
        with open(os.path.join(metrics_dir, f"som_{size_name}_error_metrics.json"), "r") as f:
            errors = json.load(f)
        qe = errors["quantization_error"]
        te = errors["topographic_error"]
        
        som_results.append({
            "Modelo": f"SOM {size_name}",
            "ARI": ari,
            "NMI": nmi,
            "Silhouette": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Pureza Neurônios": purity,
            "Erro Quantização": qe,
            "Erro Topográfico": te
        })
        
    # Baseline 1: K-Means directly on scaled data (K=6)
    print("Evaluating Baseline: K-Means (K=6)...")
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    
    ari_km = adjusted_rand_score(y, km_labels)
    nmi_km = normalized_mutual_info_score(y, km_labels)
    sil_km = silhouette_score(X_scaled, km_labels)
    db_km = davies_bouldin_score(X_scaled, km_labels)
    ch_km = calinski_harabasz_score(X_scaled, km_labels)
    
    som_results.append({
        "Modelo": "K-Means (Direto)",
        "ARI": ari_km,
        "NMI": nmi_km,
        "Silhouette": sil_km,
        "Davies-Bouldin": db_km,
        "Calinski-Harabasz": ch_km,
        "Pureza Neurônios": np.nan,
        "Erro Quantização": np.nan,
        "Erro Topográfico": np.nan
    })
    
    # Baseline 2: PCA + K-Means (K=6)
    print("Evaluating Baseline: PCA + K-Means (K=6)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    km_pca = KMeans(n_clusters=6, random_state=42, n_init=10)
    km_pca_labels = km_pca.fit_predict(X_pca)
    
    ari_pca = adjusted_rand_score(y, km_pca_labels)
    nmi_pca = normalized_mutual_info_score(y, km_pca_labels)
    sil_pca = silhouette_score(X_scaled, km_pca_labels)
    db_pca = davies_bouldin_score(X_scaled, km_pca_labels)
    ch_pca = calinski_harabasz_score(X_scaled, km_pca_labels)
    
    som_results.append({
        "Modelo": "PCA + K-Means",
        "ARI": ari_pca,
        "NMI": nmi_pca,
        "Silhouette": sil_pca,
        "Davies-Bouldin": db_pca,
        "Calinski-Harabasz": ch_pca,
        "Pureza Neurônios": np.nan,
        "Erro Quantização": np.nan,
        "Erro Topográfico": np.nan
    })
    
    # Baseline 3: Agglomerative Clustering (K=6)
    print("Evaluating Baseline: Agglomerative (K=6)...")
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=6)
    agg_labels = agg.fit_predict(X_scaled)
    
    ari_agg = adjusted_rand_score(y, agg_labels)
    nmi_agg = normalized_mutual_info_score(y, agg_labels)
    sil_agg = silhouette_score(X_scaled, agg_labels)
    db_agg = davies_bouldin_score(X_scaled, agg_labels)
    ch_agg = calinski_harabasz_score(X_scaled, agg_labels)
    
    som_results.append({
        "Modelo": "Agglomerative (K=6)",
        "ARI": ari_agg,
        "NMI": nmi_agg,
        "Silhouette": sil_agg,
        "Davies-Bouldin": db_agg,
        "Calinski-Harabasz": ch_agg,
        "Pureza Neurônios": np.nan,
        "Erro Quantização": np.nan,
        "Erro Topográfico": np.nan
    })
    
    # Baseline 4: DBSCAN
    print("Evaluating Baseline: DBSCAN...")
    from sklearn.cluster import DBSCAN
    best_sil = -1
    best_labels = None
    for eps_val in np.linspace(2.5, 7.5, 21):
        dbscan = DBSCAN(eps=eps_val, min_samples=4)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        unique_labels = set(dbscan_labels) - {-1}
        if len(unique_labels) > 1:
            try:
                sil_val = silhouette_score(X_scaled, dbscan_labels)
                if sil_val > best_sil:
                    best_sil = sil_val
                    best_labels = dbscan_labels
            except Exception:
                pass
                
    if best_labels is None:
        dbscan = DBSCAN(eps=5.0, min_samples=4)
        best_labels = dbscan.fit_predict(X_scaled)
        
    ari_db = adjusted_rand_score(y, best_labels)
    nmi_db = normalized_mutual_info_score(y, best_labels)
    sil_db = silhouette_score(X_scaled, best_labels)
    db_db = davies_bouldin_score(X_scaled, best_labels)
    ch_db = calinski_harabasz_score(X_scaled, best_labels)
    
    som_results.append({
        "Modelo": "DBSCAN",
        "ARI": ari_db,
        "NMI": nmi_db,
        "Silhouette": sil_db,
        "Davies-Bouldin": db_db,
        "Calinski-Harabasz": ch_db,
        "Pureza Neurônios": np.nan,
        "Erro Quantização": np.nan,
        "Erro Topográfico": np.nan
    })
    
    # Save comparison dataframe
    df_comparison = pd.DataFrame(som_results)
    df_comparison.to_csv(os.path.join(metrics_dir, "model_comparison_results.csv"), index=False)
    df_comparison.to_excel(os.path.join(metrics_dir, "model_comparison_results.xlsx"), index=False)
    
    print("\n=======================================================")
    print("COMPARATIVO FINAL DE MODELOS E BASELINES:")
    print("=======================================================")
    print(df_comparison.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
