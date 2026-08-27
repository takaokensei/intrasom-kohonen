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

    def build_expanded_umatrix_grid(som, umat_expanded):
        cols, rows = som.mapsize
        lattice = getattr(som.codebook, 'lattice', 'hexa')
        is_toroid = (som.mapshape == 'toroid')
        edges = []
        seen_pairs = set()

        if lattice == 'rect':
            # 8 vizinhos para malha retangular (ordem síncrona com IntraSOM 1.1.1 build_umatrix):
            # [right, down-right, down, down-left, left, up-left, up, up-right]
            rect_offsets = [
                (1, 0),    # 0 = right
                (1, 1),    # 1 = down-right
                (0, 1),    # 2 = down
                (-1, 1),   # 3 = down-left
                (-1, 0),   # 4 = left
                (-1, -1),  # 5 = up-left
                (0, -1),   # 6 = up
                (1, -1),   # 7 = up-right
            ]
            for r in range(rows):
                for c in range(cols):
                    neuron_idx = r * cols + c
                    for k in range(8):
                        dist = umat_expanded[r, c, k]
                        if np.isnan(dist):
                            continue
                        dc, dr = rect_offsets[k]
                        nr = r + dr
                        nc = c + dc
                        if is_toroid:
                            nr %= rows
                            nc %= cols
                        elif not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        neighbor_idx = nr * cols + nc
                        if neuron_idx == neighbor_idx:
                            continue
                        p1, p2 = min(neuron_idx, neighbor_idx), max(neuron_idx, neighbor_idx)
                        pair = (p1, p2)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            edges.append({"from": p1 + 1, "to": p2 + 1, "distance": float(dist)})
        else:
            # 6 vizinhos para malha hexagonal
            ii = [[1, 1, 0, -1, 0, 1], [1, 0, -1, -1, -1, 0]]
            jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]
            for r in range(rows):
                for c in range(cols):
                    neuron_idx = r * cols + c
                    e = 1 if r % 2 == 0 else 0
                    for k in range(6):
                        dist = umat_expanded[r, c, k]
                        if np.isnan(dist):
                            continue
                        nr = r + jj[e][k]
                        nc = c + ii[e][k]
                        if is_toroid:
                            nr %= rows
                            nc %= cols
                        elif not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        neighbor_idx = nr * cols + nc
                        if neuron_idx == neighbor_idx:
                            continue
                        p1, p2 = min(neuron_idx, neighbor_idx), max(neuron_idx, neighbor_idx)
                        pair = (p1, p2)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            edges.append({"from": p1 + 1, "to": p2 + 1, "distance": float(dist)})

        edge_dists = [e["distance"] for e in edges]
        min_dist = float(np.min(edge_dists)) if edge_dists else 0.0
        max_dist = float(np.max(edge_dists)) if edge_dists else 1.0
        return edges, min_dist, max_dist

    def build_neurons_list(neurons_df, results_df_local, som, y_labels, is_time_series=True):
        """Constroi a lista de neuronios para o JSON do frontend."""
        cols, rows = som.mapsize
        lattice = getattr(som.codebook, 'lattice', 'hexa')
        if lattice == 'rect':
            coords = [(float(idx % cols), float(idx // cols)) for idx in range(cols * rows)]
        else:
            plot_f = PlotFactory(som)
            coords = plot_f.generate_hex_lattice(cols, rows)
        umat_expanded = som.build_umatrix(expanded=True)
        umat          = som.build_umatrix(expanded=False)
        edges, edge_min, edge_max = build_expanded_umatrix_grid(som, umat_expanded)

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
        return {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list,
            "umatrix_edges": edges,
            "umatrix_edge_min": edge_min,
            "umatrix_edge_max": edge_max
        }

    all_sizes = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]
    for size_name in all_sizes:
        print(f"Processing SOM {size_name} (HEX_toroid, HEX_planar, RECT_planar, RECT_toroid)...")
        som_models[size_name] = {"has_variants": True}

        # Helper: carrega variante IntraSOM de arquivo parquet
        def _load_variant(nf, rf, pf):
            if not (os.path.exists(nf) and os.path.exists(rf) and os.path.exists(pf)):
                return None
            ndf = pd.read_parquet(nf)
            rdf = pd.read_parquet(rf)
            p   = json.load(open(pf))
            s   = intrasom.SOMFactory.load_som(data=X, trained_neurons=ndf, params=p)
            return build_neurons_list(ndf, rdf, s, y)

        # 1. HEX_toroid
        result = _load_variant(
            os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet"),
            os.path.join(maps_dir, f"SOM_{size_name}_results.parquet"),
            os.path.join(maps_dir, f"params_SOM_{size_name}.json"),
        )
        if result:
            som_models[size_name]["HEX_toroid"] = result

        # 2. HEX_planar
        result = _load_variant(
            os.path.join(maps_dir, f"SOM_{size_name}_HEX_planar_neurons.parquet"),
            os.path.join(maps_dir, f"SOM_{size_name}_HEX_planar_results.parquet"),
            os.path.join(maps_dir, f"params_SOM_{size_name}_HEX_planar.json"),
        )
        if result:
            som_models[size_name]["HEX_planar"] = result

        # 3. RECT_planar (IntraSOM 1.1.1 — lattice='rect', mapshape='planar')
        result = _load_variant(
            os.path.join(maps_dir, f"SOM_{size_name}_RECT_planar_neurons.parquet"),
            os.path.join(maps_dir, f"SOM_{size_name}_RECT_planar_results.parquet"),
            os.path.join(maps_dir, f"params_SOM_{size_name}_RECT_planar.json"),
        )
        if result:
            som_models[size_name]["RECT_planar"] = result

        # 4. RECT_toroid (IntraSOM 1.1.1 — lattice='rect', mapshape='toroid')
        result = _load_variant(
            os.path.join(maps_dir, f"SOM_{size_name}_RECT_toroid_neurons.parquet"),
            os.path.join(maps_dir, f"SOM_{size_name}_RECT_toroid_results.parquet"),
            os.path.join(maps_dir, f"params_SOM_{size_name}_RECT_toroid.json"),
        )
        if result:
            som_models[size_name]["RECT_toroid"] = result

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
    
    # Nota: text_rect_models.json (MiniSom legado) não é mais necessário.
    # Os modelos RECT de texto agora são carregados de parquets IntraSOM abaixo.
        
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

        # Generate visual coordinates depending on lattice type.
        # RECT: pure integer (col, row) — no hex zig-zag offset.
        # HEX : standard odd-r offset via PlotFactory.generate_hex_lattice.
        if getattr(som.codebook, 'lattice', 'hexa') == 'rect':
            coords = [(float(idx % cols), float(idx // cols)) for idx in range(cols * rows)]
        else:
            from intrasom.visualization import PlotFactory
            plot_f = PlotFactory(som)
            coords = plot_f.generate_hex_lattice(cols, rows)

        umat_expanded = som.build_umatrix(expanded=True)
        umat          = som.build_umatrix(expanded=False)
        edges, edge_min, edge_max = build_expanded_umatrix_grid(som, umat_expanded)
        
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

            if bmu_idx in counts.index and bmu_totals > 0:
                probs = counts.loc[bmu_idx].values / bmu_totals
                probs = probs[probs > 0]
                bmu_entropy = float(-np.sum(probs * np.log2(probs)))
            else:
                bmu_entropy = 0.0
            
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
                "entropy": bmu_entropy,
                "total_samples": bmu_totals,
                "doc_indices": doc_indices,
                "codebook": codebook
            })
            
        return {
            "cols": cols,
            "rows": rows,
            "neurons": neurons_list,
            "umatrix_edges": edges,
            "umatrix_edge_min": edge_min,
            "umatrix_edge_max": edge_max

        }

    for dname, dinfo in datasets_info.items():
        text_models[dname] = {}
        docs, text_labels = dinfo["load_fn"]()
        num_docs = len(docs)
        
        for rep in ["TF-IDF", "SBERT", "BGE-M3", "Gemma-300M"]:
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

            # 3. RECT_planar (IntraSOM 1.1.1 — lattice='rect', mapshape='planar')
            rect_planar_nf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_RECT_planar_neurons.parquet")
            rect_planar_rf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_RECT_planar_results.parquet")
            rect_planar_pf = os.path.join(maps_dir, f"params_SOM_Text_{dname}_{rep}_RECT_planar.json")
            rect_planar_model = build_text_som_model_from_files(
                rect_planar_nf, rect_planar_rf, rect_planar_pf, text_labels, num_docs
            )

            # 4. RECT_toroid (IntraSOM 1.1.1 — lattice='rect', mapshape='toroid')
            rect_toroid_nf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_RECT_toroid_neurons.parquet")
            rect_toroid_rf = os.path.join(maps_dir, f"SOM_Text_{dname}_{rep}_RECT_toroid_results.parquet")
            rect_toroid_pf = os.path.join(maps_dir, f"params_SOM_Text_{dname}_{rep}_RECT_toroid.json")
            rect_toroid_model = build_text_som_model_from_files(
                rect_toroid_nf, rect_toroid_rf, rect_toroid_pf, text_labels, num_docs
            )

            # Assemble variants dict (mirrors SCM pattern) — 4 variantes reais
            variant_dict: dict = {"has_variants": True, "HEX_toroid": hex_toroid_model}
            if hex_planar_model:
                variant_dict["HEX_planar"] = hex_planar_model
            if rect_planar_model:
                variant_dict["RECT_planar"] = rect_planar_model
            if rect_toroid_model:
                variant_dict["RECT_toroid"] = rect_toroid_model

            text_models[dname][rep] = variant_dict

            
    with open(os.path.join(public_data_dir, "text_models.json"), "w", encoding="utf-8") as f:
        json.dump(text_models, f, ensure_ascii=False)
    print("Exported text_models.json")
    
    # 5. Export PCA parameters for ALL embedding models (SBERT, BGE-M3, Gemma-300M)
    pca_params_combined = {}
    for dname in ["20news", "6class"]:
        pca_params_combined[dname] = {}
        for rep_key, pkl_prefix in [("SBERT", "sbert"), ("BGE-M3", "bgem3"), ("Gemma-300M", "gemma300m")]:
            pkl_path = os.path.join(maps_dir, f"{dname}_{pkl_prefix}_pca.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        pca = pickle.load(f)
                    pca_params_combined[dname][rep_key] = {
                        "mean": pca.mean_.tolist(),
                        "components": pca.components_.tolist()
                    }
                except Exception as e:
                    print(f"Error loading PCA for {dname} {rep_key}: {e}")
            
    with open(os.path.join(public_data_dir, "pca_params.json"), "w", encoding="utf-8") as f:
        json.dump(pca_params_combined, f, ensure_ascii=False)
    print("Exported pca_params.json")
        
    # 6. Text metrics
    text_metrics_file = os.path.join(metrics_dir, "text_clustering_comparison.json")
    if os.path.exists(text_metrics_file):
        text_metrics = json.load(open(text_metrics_file))
    else:
        text_metrics = {}

    # Merge new embedding metrics if available
    new_metrics_file = os.path.join(metrics_dir, "new_embeddings_metrics.json")
    if os.path.exists(new_metrics_file):
        new_metrics_list = json.load(open(new_metrics_file))
        for item in new_metrics_list:
            d_name = item.get("dataset_name")
            r_name = item.get("representation_name")
            v_name = item.get("variant")
            if v_name == "HEX_toroid" and d_name and r_name:
                if d_name not in text_metrics:
                    text_metrics[d_name] = {}
                text_metrics[d_name][r_name] = {
                    "ARI": item.get("ARI", 0.0),
                    "NMI": item.get("NMI", 0.0)
                }
                # Also support underscore version for backwards compatibility
                text_metrics[d_name][r_name.replace("-", "_")] = {
                    "ARI": item.get("ARI", 0.0),
                    "NMI": item.get("NMI", 0.0)
                }
        
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
