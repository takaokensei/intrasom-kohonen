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
    # ─ Exporta sub-chaves de variante (HEX_toroid, HEX_planar, RECT_planar) para TODOS os 6 tamanhos
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

    # Carregar modelos MiniSom RECT pré-gerados
    rect_models_path = os.path.join(maps_dir, "som_rect_models.json")
    rect_models_data = json.load(open(rect_models_path, "r", encoding="utf-8")) if os.path.exists(rect_models_path) else {}

    all_sizes = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]
    for size_name in all_sizes:
        print(f"Processing SOM {size_name} (HEX_toroid, HEX_planar, RECT_planar)...")
        som_models[size_name] = {"has_variants": True}

        # 1. HEX_toroid
        toroid_nf = os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet")
        toroid_rf = os.path.join(maps_dir, f"SOM_{size_name}_results.parquet")
        toroid_pf = os.path.join(maps_dir, f"params_SOM_{size_name}.json")
        if os.path.exists(toroid_nf) and os.path.exists(toroid_rf) and os.path.exists(toroid_pf):
            neurons_df = pd.read_parquet(toroid_nf)
            results_df = pd.read_parquet(toroid_rf)
            params     = json.load(open(toroid_pf))
            som        = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
            neurons_list, cols, rows = build_neurons_list(neurons_df, results_df, som, y)
            som_models[size_name]["HEX_toroid"] = {"cols": cols, "rows": rows, "neurons": neurons_list}

        # 2. HEX_planar
        planar_nf = os.path.join(maps_dir, f"SOM_{size_name}_HEX_planar_neurons.parquet")
        planar_rf = os.path.join(maps_dir, f"SOM_{size_name}_HEX_planar_results.parquet")
        planar_pf = os.path.join(maps_dir, f"params_SOM_{size_name}_HEX_planar.json")
        if os.path.exists(planar_nf) and os.path.exists(planar_rf) and os.path.exists(planar_pf):
            neurons_df = pd.read_parquet(planar_nf)
            results_df = pd.read_parquet(planar_rf)
            params     = json.load(open(planar_pf))
            som        = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
            neurons_list, cols, rows = build_neurons_list(neurons_df, results_df, som, y)
            som_models[size_name]["HEX_planar"] = {"cols": cols, "rows": rows, "neurons": neurons_list}

        # 3. RECT_planar (MiniSom)
        if size_name in rect_models_data:
            som_models[size_name]["RECT_planar"] = rect_models_data[size_name]

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
    from text_data import load_20news_data, load_6class_data
    
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
    
    # Carregar modelos MiniSom RECT de texto pré-gerados por train_text_som_rect.py
    text_rect_path = os.path.join(maps_dir, "text_rect_models.json")
    text_rect_data = json.load(open(text_rect_path, "r", encoding="utf-8")) if os.path.exists(text_rect_path) else {}
    if not text_rect_data:
        print("WARNING: text_rect_models.json not found. Run src/train_text_som_rect.py first.")

    for dname, dinfo in datasets_info.items():
        text_models[dname] = {}
        docs, text_labels = dinfo["load_fn"]()
        num_docs = len(docs)
        
    def build_text_som_model_from_files(neurons_file, results_file, params_file, text_labels, num_docs):
        if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
            return None
            
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
        
        results_df_copy = results_df.copy()
        results_df_copy['Class'] = text_labels
        counts = results_df_copy.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
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
            
            doc_indices = [int(i.split("_")[1]) - 1 for i in results_df_copy[results_df_copy['BMU'] == bmu_idx].index]
            
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
            
        return {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list
        }

    for dname, dinfo in datasets_info.items():
        text_models[dname] = {}
        docs, text_labels = dinfo["load_fn"]()
        num_docs = len(docs)
        
        for rep in ["TF-IDF", "SBERT"]:
            # 1. HEX_toroid
            toroid_nf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_neurons.parquet")
            toroid_rf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_results.parquet")
            toroid_pf = os.path.join(maps_dir, f"params_SOM_Text_{dname}_{rep}.json")
            hex_toroid_model = build_text_som_model_from_files(toroid_nf, toroid_rf, toroid_pf, text_labels, num_docs)

            if not hex_toroid_model:
                print(f"Text SOM {dname} {rep} HEX_toroid files not found. Skipping.")
                continue

            # 2. HEX_planar
            planar_nf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_HEX_planar_neurons.parquet")
            planar_rf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_HEX_planar_results.parquet")
            planar_pf = os.path.join(maps_dir, f"params_SOM_Text_{dname}_{rep}_HEX_planar.json")
            hex_planar_model = build_text_som_model_from_files(planar_nf, planar_rf, planar_pf, text_labels, num_docs)

            # 3. RECT_planar
            rect_model = text_rect_data.get(dname, {}).get(rep, {}).get("RECT_planar", None)

            # Assemble variants dict (mirrors SCM pattern)
            variant_dict = {
                "has_variants": True,
                "HEX_toroid": hex_toroid_model
            }
            if hex_planar_model:
                variant_dict["HEX_planar"] = hex_planar_model
            if rect_model:
                variant_dict["RECT_planar"] = rect_model

            text_models[dname][rep] = variant_dict

            
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
