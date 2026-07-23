import os, sys, shutil, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
import intrasom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAP_SIZES = {
    "5x5":   (5, 5),
    "7x7":   (7, 7),
    "10x10": (10, 10),
    "12x12": (12, 12),
    "15x15": (15, 15),
    "20x20": (20, 20),
}

def train_variant(X, size_name, mapsize):
    name = f"SOM_{size_name}_HEX_planar"
    tp = compute_train_params(mapsize)
    print(f"Training Planar SOM for {size_name} {mapsize}...")
    set_global_seed()
    som = intrasom.SOMFactory.build(
        data=X,
        mapsize=mapsize,
        mapshape="planar",
        lattice="hexa",
        normalization="var",
        initialization="pca",
        neighborhood="gaussian",
        training="batch",
        name=name,
        sample_names=list(X.index)
    )
    som.train(previous_epoch=True, **tp)
    qe = som.calculate_quantization_error
    if callable(qe): qe = qe()
    te = som.topographic_error
    if callable(te): te = te()
    print(f"  {size_name} HEX_planar - QE: {qe:.4f}, TE: {te:.4f}")
    
    src_dir = os.path.join(os.getcwd(), "Results")
    if os.path.exists(src_dir):
        for fn in os.listdir(src_dir):
            if name in fn:
                s = os.path.join(src_dir, fn)
                d = os.path.join(MAPS_DIR, fn)
                if os.path.exists(d): os.remove(d)
                shutil.move(s, d)
                
    return {
        "size_name": size_name,
        "variant_key": "HEX_planar",
        "mapsize": list(mapsize),
        "lattice": "hexa",
        "mapshape": "planar",
        "initialization": "pca",
        "total_epochs": TOTAL_EPOCHS,
        "train_rough_len": tp["train_rough_len"],
        "train_finetune_len": tp["train_finetune_len"],
        "train_rough_radiusin": tp["train_rough_radiusin"],
        "train_rough_radiusfin": RADIUS_FINAL,
        "quantization_error": float(qe),
        "topographic_error": float(te)
    }

def main():
    X, y = load_synthetic_control()
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    results = []
    for size_name, mapsize in MAP_SIZES.items():
        res = train_variant(X, size_name, mapsize)
        results.append(res)
        
    out = os.path.join(METRICS_DIR, "som_variants_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved Planar variants metrics to " + out)

if __name__ == "__main__":
    main()

