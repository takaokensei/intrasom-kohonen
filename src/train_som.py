import os
import shutil
import json
import pandas as pd
import numpy as np
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control

import intrasom

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
        
        # Build SOM
        som = intrasom.SOMFactory.build(
            data=X,
            mapsize=mapsize,
            mapshape='toroid',
            lattice='hexa',
            normalization='var',
            initialization='random',
            neighborhood='gaussian',
            training='batch',
            name=f"SOM_{size_name}",
            sample_names=list(X.index)
        )
        
        # Train SOM
        som.train(train_len_factor=2, previous_epoch=True)
        
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
            "size_name": size_name,
            "mapsize": mapsize,
            "quantization_error": float(qe),
            "topographic_error": float(te)
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
