import os
import sys
import json
import pandas as pd
import numpy as np
import pickle

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
        
    # 2. SOM models data
    # ─ 10x10: exportado com sub-chaves de variante (HEX_toroid, HEX_planar)
    # ─ 15x15, 20x20: estrutura flat para compatibilidade retroativa
    som_models = {}
    from intrasom.visualization import PlotFactory

    def build_neurons_list(neurons_df, results_df_local, som, y_labels, is_time_series=True):
        """Constroi a lista de neuronios para o JSON do frontend."""
        cols, rows = som.mapsize
        plot_f     = PlotFactory(som)
        coords     = plot_f.generate_hex_lattice(cols, rows)
        umat       = som.build_umatrix(expanded=False)
        results_df_local = results_df_local.copy()
        results_df_local['Class'] = y_labels
        counts        = results_df_local.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
        totals        = counts.sum(axis=1)
        dominant_class = counts.idxmax(axis=1)
        purity        = counts.max(axis=1) / totals
        neurons_list = []
        for idx in range(cols * rows):
            bmu_idx = idx + 1
            cx, cy  = coords[idx]
            if is_time_series:
                weight_cols = [c for c in neurons_df.columns if c.startswith("B_Time_")]
                weight_cols = sorted(weight_cols, key=lambda c: int(c.split("_")[-1]))
                codebook = neurons_df.loc[bmu_idx, weight_cols].values.tolist() if bmu_idx in neurons_df.index else []
            else:
                codebook = []
            bmu_totals  = int(totals.get(bmu_idx, 0))
            bmu_dominant = dominant_class.get(bmu_idx, "Nenhum") if bmu_totals > 0 else "Vazio"
            bmu_purity   = float(purity.get(bmu_idx, 0))
            sample_ids   = [int(i.split("_")[1]) for i in results_df_local[results_df_local['BMU'] == bmu_idx].index]
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
                "codebook": codebook,
            })
        return neurons_list, cols, rows

    # ── 10x10: exportar variantes HEX_toroid e HEX_planar ────────────────────
    print("Processing SOM 10x10 (with variants HEX_toroid + HEX_planar)...")
    toroid_nf  = os.path.join(maps_dir, "SOM_10x10_neurons.parquet")
    toroid_rf  = os.path.join(maps_dir, "SOM_10x10_results.parquet")
    toroid_pf  = os.path.join(maps_dir, "params_SOM_10x10.json")
    planar_nf  = os.path.join(maps_dir, "SOM_10x10_HEX_planar_neurons.parquet")
    planar_rf  = os.path.join(maps_dir, "SOM_10x10_HEX_planar_results.parquet")
    planar_pf  = os.path.join(maps_dir, "params_SOM_10x10_HEX_planar.json")

    som_models["10x10"] = {"has_variants": True}

    for vkey, nf, rf, pf in [
        ("HEX_toroid", toroid_nf, toroid_rf, toroid_pf),
        ("HEX_planar", planar_nf, planar_rf, planar_pf),
    ]:
        if not (os.path.exists(nf) and os.path.exists(rf) and os.path.exists(pf)):
            print(f"  Variant {vkey} files not found. Skipping.")
            continue
        neurons_df = pd.read_parquet(nf)
        results_df = pd.read_parquet(rf)
        params     = json.load(open(pf))
        som        = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
        neurons_list, cols, rows = build_neurons_list(neurons_df, results_df, som, y)
        som_models["10x10"][vkey] = {"cols": cols, "rows": rows, "neurons": neurons_list}
        print(f"  Variant {vkey}: {cols}x{rows}, {len(neurons_list)} neurons")

    # ── 15x15, 20x20: estrutura flat (compatibilidade retroativa) ─────────────
    for size_name in ["15x15", "20x20"]:
        print(f"Processing SOM {size_name}...")
        neurons_file = os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet")
        results_file = os.path.join(maps_dir, f"SOM_{size_name}_results.parquet")
        params_file  = os.path.join(maps_dir, f"params_SOM_{size_name}.json")
        if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
            print(f"SOM {size_name} files not found. Skipping.")
            continue
        neurons_df = pd.read_parquet(neurons_file)
        results_df = pd.read_parquet(results_file)
        params     = json.load(open(params_file))
        som        = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
        neurons_list, cols, rows = build_neurons_list(neurons_df, results_df, som, y)
        som_models[size_name] = {"cols": cols, "rows": rows, "neurons": neurons_list}

    with open(os.path.join(public_data_dir, "som_models.json"), "w", encoding="utf-8") as f:
        json.dump(som_models, f, ensure_ascii=False)
    print("Exported som_models.json")
        
    # 3. Model Comparison Metrics
    print("Loading metrics comparison...")
    metrics_file = os.path.join(metrics_dir, "model_comparison_results.csv")
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        metrics_df = metrics_df.replace({np.nan: None})
        metrics_list = metrics_df.to_dict(orient="records")
    else:
        metrics_list = []
        
    with open(os.path.join(public_data_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False)
    print("Exported metrics.json")
        
    # 4. Text SOM models (TF-IDF vs SBERT) for both datasets
    print("Loading text SOM results for both datasets...")
    text_models = {}
    from text_som_clustering import load_20news_data, load_6class_data
    
    datasets_info = {
        "20news": {
            "load_fn": load_20news_data,
            "classes_count": 4,
            "dims_count": 20
        },
        "6class": {
            "load_fn": load_6class_data,
            "classes_count": 6,
            "dims_count": 20
        }
    }
    
    for dname, dinfo in datasets_info.items():
        text_models[dname] = {}
        docs, text_labels = dinfo["load_fn"]()
        num_docs = len(docs)
        
        for rep in ["TF-IDF", "SBERT"]:
            neurons_file = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_neurons.parquet")
            results_file = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_results.parquet")
            params_file = os.path.join(maps_dir, f"params_SOM_Text_{dname}_{rep}.json")
            
            if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
                print(f"Text SOM {dname} {rep} files not found. Skipping.")
                continue
                
            neurons_df = pd.read_parquet(neurons_file)
            results_df = pd.read_parquet(results_file)
            params = json.load(open(params_file))
            
            dummy_data = pd.DataFrame(np.zeros((num_docs, 20)))
            dummy_data.columns = [f"Dim_{i+1}" for i in range(20)]
            dummy_data.index = [f"Doc_{i+1}" for i in range(num_docs)]
            
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
                
            text_models[dname][rep] = {
                "cols": cols,
                "rows": rows,
                "neurons": neurons_list
            }
            
    with open(os.path.join(public_data_dir, "text_models.json"), "w", encoding="utf-8") as f:
        json.dump(text_models, f, ensure_ascii=False)
    print("Exported text_models.json")
    
    # 5. Export PCA parameters for BOTH SBERT models
    pca_params_combined = {}
    for dname in ["20news", "6class"]:
        try:
            with open(os.path.join(maps_dir, f"{dname}_sbert_pca.pkl"), "rb") as f:
                pca = pickle.load(f)
            pca_params_combined[dname] = {
                "mean": pca.mean_.tolist(),
                "components": pca.components_.tolist()
            }
        except Exception as e:
            print(f"Error loading PCA for {dname}: {e}")
            
    with open(os.path.join(public_data_dir, "pca_params.json"), "w", encoding="utf-8") as f:
        json.dump(pca_params_combined, f, ensure_ascii=False)
    print("Exported pca_params.json")
        
    # 6. Text metrics
    text_metrics_file = os.path.join(metrics_dir, "text_clustering_comparison.json")
    if os.path.exists(text_metrics_file):
        text_metrics = json.load(open(text_metrics_file))
    else:
        text_metrics = {}
        
    with open(os.path.join(public_data_dir, "text_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(text_metrics, f, ensure_ascii=False)
    print("Exported text_metrics.json")
        
    # 7. News samples for both datasets
    news_samples_combined = {}
    
    # 20news samples
    docs_20news, labels_20news = load_20news_data()
    news_samples_combined["20news"] = []
    for cat in ["Graphics", "Space", "Baseball", "Mideast"]:
        indices = np.where(labels_20news == cat)[0][:10]
        for idx in indices:
            news_samples_combined["20news"].append({
                "id": int(idx),
                "text": docs_20news[idx][:350] + ("..." if len(docs_20news[idx]) > 350 else ""),
                "class": labels_20news[idx]
            })
            
    # 6class samples
    docs_6class, labels_6class = load_6class_data()
    news_samples_combined["6class"] = []
    for cat in ["Turismo", "Esportes", "Policia", "Economia", "Politica", "Variedades"]:
        indices = np.where(labels_6class == cat)[0][:10]
        for idx in indices:
            news_samples_combined["6class"].append({
                "id": int(idx),
                "text": docs_6class[idx][:350] + ("..." if len(docs_6class[idx]) > 350 else ""),
                "class": labels_6class[idx]
            })
            
    with open(os.path.join(public_data_dir, "news_samples.json"), "w", encoding="utf-8") as f:
        json.dump(news_samples_combined, f, ensure_ascii=False)
    print("Exported news_samples.json")

    # 8. Parameter study results (for ParameterStudyPanel in dashboard)
    param_study_file = os.path.join(metrics_dir, "parameter_study.json")
    if os.path.exists(param_study_file):
        param_study = json.load(open(param_study_file))
    else:
        param_study = []
        print("WARNING: parameter_study.json not found. Run src/train_parameter_study.py first.")
    with open(os.path.join(public_data_dir, "parameter_study.json"), "w", encoding="utf-8") as f:
        json.dump(param_study, f, ensure_ascii=False)
    print("Exported parameter_study.json")

    # Clean temporary files if any
    ts_file = os.path.join(workspace_dir, "frontend", "src", "data", "precomputed_data.ts")
    if os.path.exists(ts_file):
        os.remove(ts_file)
        
    print("All frontend data successfully exported to public/data/")

if __name__ == "__main__":
    export_all()
