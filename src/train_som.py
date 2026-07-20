import os
import shutil
import json
import pandas as pd
import numpy as np
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed

import intrasom

# ── Parâmetros de treino recomendados pelo Prof. José Alfredo ──────────────
# Razão rough/finetune: 30%/70% das 500 épocas totais.
# Fase rough: raio grande → organização topológica global do mapa.
# Fase finetune: raio 1 → ajuste fino preciso das vizinhanças locais.
TOTAL_EPOCHS      = 500   # total de épocas (rough + finetune)
ROUGH_FRACTION    = 0.30  # 30% das épocas na fase rough
RADIUS_INIT_FRAC  = 0.80  # raio inicial = 80% do maior lado do mapa
RADIUS_FINAL      = 1     # raio final = 1 neurônio (ajuste fino máximo)


def compute_train_params(mapsize: tuple) -> dict:
    """Calcula os parâmetros de treino explícitos a partir do tamanho do mapa.

    Garante que a soma rough + finetune seja sempre TOTAL_EPOCHS e que o
    raio inicial seja exatamente 80% do maior lado do mapa, conforme
    instrução do professor.
    """
    rough_len    = round(TOTAL_EPOCHS * ROUGH_FRACTION)         # 150
    finetune_len = TOTAL_EPOCHS - rough_len                     # 350
    radius_in    = max(1, round(RADIUS_INIT_FRAC * max(mapsize)))  # e.g. 8 para 10×10
    return {
        "train_rough_len":          rough_len,
        "train_rough_radiusin":     radius_in,
        "train_rough_radiusfin":    RADIUS_FINAL,
        "train_finetune_len":       finetune_len,
        "train_finetune_radiusin":  RADIUS_FINAL,
        "train_finetune_radiusfin": RADIUS_FINAL,
    }

def train_all_soms():
    # 1. Load data
    print("Loading Synthetic Control dataset...")
    X, y = load_synthetic_control()
    
    # 2. Define map sizes to test
    map_sizes = {
        "5x5": (5, 5),
        "7x7": (7, 7),
        "10x10": (10, 10),
        "12x12": (12, 12),
        "15x15": (15, 15),
        "20x20": (20, 20)
    }
    
    # Prepare directories
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(workspace_dir, "outputs", "maps"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "outputs", "metrics"), exist_ok=True)
    
    summary_results = []
    
    for size_name, mapsize in map_sizes.items():
        print(f"\n==========================================")
        print(f"Training SOM size: {size_name} {mapsize}")
        print(f"==========================================")

        train_params = compute_train_params(mapsize)
        print(f"  initialization : pca")
        print(f"  rough epochs   : {train_params['train_rough_len']}")
        print(f"  finetune epochs: {train_params['train_finetune_len']}")
        print(f"  total epochs   : {TOTAL_EPOCHS}")
        print(f"  radius in      : {train_params['train_rough_radiusin']} "
              f"({RADIUS_INIT_FRAC*100:.0f}% of max({mapsize})={max(mapsize)})")
        print(f"  radius fin     : {RADIUS_FINAL}")

        # Seed fixada imediatamente antes do build (intrasom usa np.random global)
        set_global_seed()

        # Build SOM — initialization='pca' conforme instrução do professor
        som = intrasom.SOMFactory.build(
            data=X,
            mapsize=mapsize,
            mapshape='toroid',
            lattice='hexa',
            normalization='var',
            initialization='pca',     # era 'random' → PCA conforme instrução
            neighborhood='gaussian',
            training='batch',
            name=f"SOM_{size_name}",
            sample_names=list(X.index)
        )

        # Train SOM com parâmetros explícitos de época e raio
        som.train(previous_epoch=True, **train_params)
        
        # Access errors
        # Note: calculate_quantization_error might be a property or a method, let's try reading it.
        # Let's inspect if it is callable.
        qe = som.calculate_quantization_error
        if callable(qe):
            qe = qe()
        
        te = som.topographic_error
        if callable(te):
            te = te()
            
        print(f"SOM {size_name} - Quantization Error: {qe:.4f}, Topographic Error: {te:.4f}")
        
        # Save training summary info
        model_info = {
            "size_name":             size_name,
            "mapsize":               list(mapsize),
            "quantization_error":    float(qe),
            "topographic_error":     float(te),
            "initialization":        "pca",
            "total_epochs":          TOTAL_EPOCHS,
            "train_rough_len":       train_params["train_rough_len"],
            "train_finetune_len":    train_params["train_finetune_len"],
            "train_rough_radiusin":  train_params["train_rough_radiusin"],
            "train_rough_radiusfin": RADIUS_FINAL,
        }
        
        summary_results.append(model_info)
        
        # Move generated result files to outputs/maps
        results_src_dir = os.path.join(os.getcwd(), "Results")
        if os.path.exists(results_src_dir):
            for file_name in os.listdir(results_src_dir):
                if file_name.startswith(f"SOM_{size_name}") or file_name.endswith(f"SOM_{size_name}.txt"):
                    src_file = os.path.join(results_src_dir, file_name)
                    dest_file = os.path.join(workspace_dir, "outputs", "maps", file_name)
                    shutil.move(src_file, dest_file)
                    
        # Let's write the specific model metrics to JSON
        with open(os.path.join(workspace_dir, "outputs", "metrics", f"som_{size_name}_error_metrics.json"), "w") as f:
            json.dump(model_info, f, indent=4)
            
    # Save overall summary metrics
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(workspace_dir, "outputs", "metrics", "som_error_comparison.csv"), index=False)
    print("\nTraining of all SOMs completed successfully. Error metrics summary:")
    print(summary_df)

if __name__ == "__main__":
    train_all_soms()
