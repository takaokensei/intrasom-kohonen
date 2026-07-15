import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
import intrasom

def export_all():
    print("Starting data export for React frontend static files...")
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(workspace_dir, "outputs", "maps")
    metrics_dir = os.path.join(workspace_dir, "outputs", "metrics")
    
    # Destination folder
    public_data_dir = os.path.join(workspace_dir, "frontend", "public", "data")
    os.makedirs(public_data_dir, exist_ok=True)
    
    # 1. Synthetic Control raw series data
    print("Loading Synthetic Control data...")
    X, y = load_synthetic_control(os.path.join(workspace_dir, "data", "synthetic_control.data"))
    
    series_data = []
    for idx, row in X.iterrows():
        sample_num = int(idx.split("_")[1])
        series_data.append({
            "id": sample_num,
            "values": row.values.tolist(),
            "class": y[sample_num - 1]
        })
        
    with open(os.path.join(public_data_dir, "series.json"), "w", encoding="utf-8") as f:
        json.dump(series_data, f, ensure_ascii=False)
    print("Exported series.json")
        
    # 2. SOM models data (10x10, 15x15, 20x20)
    som_models = {}
    for size_name in ["10x10", "15x15", "20x20"]:
        print(f"Processing SOM {size_name}...")
        neurons_file = os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet")
        results_file = os.path.join(maps_dir, f"SOM_{size_name}_results.parquet")
        params_file = os.path.join(maps_dir, f"params_SOM_{size_name}.json")
        
        if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
            print(f"SOM {size_name} files not found. Skipping.")
            continue
            
        neurons_df = pd.read_parquet(neurons_file)
        results_df = pd.read_parquet(results_file)
        params = json.load(open(params_file))
        
        som = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
        cols, rows = som.mapsize
        
        # Get coordinates
        from intrasom.visualization import PlotFactory
        plot_f = PlotFactory(som)
        coords = plot_f.generate_hex_lattice(cols, rows)
        
        # Compute U-Matrix
        umat = som.build_umatrix(expanded=False)
        
        # Map samples to BMU
        results_df['Class'] = y
        counts = results_df.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
        totals = counts.sum(axis=1)
        dominant_class = counts.idxmax(axis=1)
        purity = counts.max(axis=1) / totals
        
        neurons_list = []
        for idx in range(cols * rows):
            bmu_idx = idx + 1
            cx, cy = coords[idx]
            
            weight_cols = [c for c in neurons_df.columns if c.startswith("B_Time_")]
            weight_cols = sorted(weight_cols, key=lambda c: int(c.split("_")[-1]))
            codebook = neurons_df.loc[bmu_idx, weight_cols].values.tolist()
            
            bmu_totals = int(totals.get(bmu_idx, 0))
            bmu_dominant = dominant_class.get(bmu_idx, "Nenhum") if bmu_totals > 0 else "Vazio"
            bmu_purity = float(purity.get(bmu_idx, 0))
            
            sample_ids = [int(idx.split("_")[1]) for idx in results_df[results_df['BMU'] == bmu_idx].index]
            
            r_idx = idx // cols
            c_idx = idx % cols
            
            neurons_list.append({
                "id": bmu_idx,
                "x": float(cx),
                "y": float(cy),
                "row": int(r_idx),
                "col": int(c_idx),
                "umatrix_value": float(umat[r_idx][c_idx]),
                "dominant_class": bmu_dominant,
                "purity": bmu_purity,
                "total_samples": bmu_totals,
                "sample_ids": sample_ids,
                "codebook": codebook
            })
            
        som_models[size_name] = {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list
        }
        
    with open(os.path.join(public_data_dir, "som_models.json"), "w", encoding="utf-8") as f:
        json.dump(som_models, f, ensure_ascii=False)
    print("Exported som_models.json")
        
    # 3. Model Comparison Metrics
    print("Loading metrics comparison...")
    metrics_file = os.path.join(metrics_dir, "model_comparison_results.csv")
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        # Handle nan values for JSON conversion
        metrics_df = metrics_df.replace({np.nan: None})
        metrics_list = metrics_df.to_dict(orient="records")
    else:
        metrics_list = []
        
    with open(os.path.join(public_data_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False)
    print("Exported metrics.json")
        
    # 4. Text SOM models (TF-IDF vs SBERT)
    print("Loading text SOM results...")
    text_models = {}
    for rep in ["TF-IDF", "SBERT"]:
        neurons_file = os.path.join(maps_dir, f"SOM_Text_{rep}_neurons.parquet")
        results_file = os.path.join(maps_dir, f"SOM_Text_{rep}_results.parquet")
        params_file = os.path.join(maps_dir, f"params_SOM_Text_{rep}.json")
        
        if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
            print(f"Text SOM {rep} files not found. Skipping.")
            continue
            
        neurons_df = pd.read_parquet(neurons_file)
        results_df = pd.read_parquet(results_file)
        params = json.load(open(params_file))
        
        from text_som_clustering import load_news_data
        docs, text_labels = load_news_data()
        
        dummy_data = pd.DataFrame(np.zeros((400, 20)))
        dummy_data.columns = [f"Dim_{i+1}" for i in range(20)]
        dummy_data.index = [f"Doc_{i+1}" for i in range(400)]
        
        som = intrasom.SOMFactory.load_som(data=dummy_data, trained_neurons=neurons_df, params=params)
        cols, rows = som.mapsize
        
        from intrasom.visualization import PlotFactory
        plot_f = PlotFactory(som)
        coords = plot_f.generate_hex_lattice(cols, rows)
        umat = som.build_umatrix(expanded=False)
        
        results_df['Class'] = text_labels
        counts = results_df.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
        totals = counts.sum(axis=1)
        dominant_class = counts.idxmax(axis=1)
        purity = counts.max(axis=1) / totals
        
        neurons_list = []
        for idx in range(cols * rows):
            bmu_idx = idx + 1
            cx, cy = coords[idx]
            
            bmu_totals = int(totals.get(bmu_idx, 0))
            bmu_dominant = dominant_class.get(bmu_idx, "Nenhum") if bmu_totals > 0 else "Vazio"
            bmu_purity = float(purity.get(bmu_idx, 0))
            
            doc_indices = [int(idx.split("_")[1]) - 1 for idx in results_df[results_df['BMU'] == bmu_idx].index]
            
            r_idx = idx // cols
            c_idx = idx % cols
            
            # Get codebook from neurons_df
            neuron_row = neurons_df[neurons_df['BMU'] == bmu_idx]
            dim_cols = [f"B_Dim_{i}" for i in range(1, 21)]
            codebook = neuron_row[dim_cols].values[0].tolist() if not neuron_row.empty else []
            
            neurons_list.append({
                "id": bmu_idx,
                "x": float(cx),
                "y": float(cy),
                "row": int(r_idx),
                "col": int(c_idx),
                "umatrix_value": float(umat[r_idx][c_idx]),
                "dominant_class": bmu_dominant,
                "purity": bmu_purity,
                "total_samples": bmu_totals,
                "doc_indices": doc_indices,
                "codebook": codebook
            })
            
        text_models[rep] = {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list
        }
        
    with open(os.path.join(public_data_dir, "text_models.json"), "w", encoding="utf-8") as f:
        json.dump(text_models, f, ensure_ascii=False)
    print("Exported text_models.json")
    
    # Export PCA parameters for client-side SBERT projection
    try:
        import pickle
        with open(os.path.join(maps_dir, "sbert_pca.pkl"), "rb") as f:
            pca = pickle.load(f)
        pca_params = {
            "mean": pca.mean_.tolist(),
            "components": pca.components_.tolist()
        }
        with open(os.path.join(public_data_dir, "pca_params.json"), "w", encoding="utf-8") as f:
            json.dump(pca_params, f, ensure_ascii=False)
        print("Exported pca_params.json")
    except Exception as e:
        print(f"Error exporting pca_params.json: {e}")
        
    # Text clustering comparison metrics
    text_metrics_file = os.path.join(metrics_dir, "text_clustering_comparison.json")
    if os.path.exists(text_metrics_file):
        text_metrics = json.load(open(text_metrics_file))
    else:
        text_metrics = {}
        
    with open(os.path.join(public_data_dir, "text_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(text_metrics, f, ensure_ascii=False)
    print("Exported text_metrics.json")
        
    # Fetch representative news docs
    from text_som_clustering import load_news_data
    docs, text_labels = load_news_data()
    news_samples = []
    for cat in ["Graphics", "Space", "Baseball", "Mideast"]:
        indices = np.where(text_labels == cat)[0][:15]
        for idx in indices:
            news_samples.append({
                "id": int(idx),
                "text": docs[idx][:350] + ("..." if len(docs[idx]) > 350 else ""),
                "class": text_labels[idx]
            })
            
    with open(os.path.join(public_data_dir, "news_samples.json"), "w", encoding="utf-8") as f:
        json.dump(news_samples, f, ensure_ascii=False)
    print("Exported news_samples.json")
    
    # Remove the large precomputed_data.ts to save typescript space
    ts_file = os.path.join(workspace_dir, "frontend", "src", "data", "precomputed_data.ts")
    if os.path.exists(ts_file):
        os.remove(ts_file)
        print("Removed precomputed_data.ts bundle file")
        
    print("All frontend data successfully exported to public/data/")

if __name__ == "__main__":
    export_all()
