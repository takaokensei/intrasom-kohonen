import os, sys, shutil, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import compute_train_params, TOTAL_EPOCHS, RADIUS_FINAL
import intrasom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

# lattice='rect' NOT supported: intrasom.py line 1928 raises build_umatrix exception.
# Dashboard will show explicit limitation badge. Only 'planar' mapshape is trained here.
VARIANTS = [("hexa", "planar", "HEX_planar")]
MAPSIZE  = (10, 10)

def train_variant(X, lattice, mapshape, variant_key):
    name = "SOM_10x10_" + variant_key
    tp = compute_train_params(MAPSIZE)
    set_global_seed()
    som = intrasom.SOMFactory.build(data=X, mapsize=MAPSIZE, mapshape=mapshape,
        lattice=lattice, normalization="var", initialization="pca",
        neighborhood="gaussian", training="batch", name=name,
        sample_names=list(X.index))
    som.train(previous_epoch=True, **tp)
    qe = som.calculate_quantization_error
    if callable(qe): qe = qe()
    te = som.topographic_error
    if callable(te): te = te()
    src_dir = os.path.join(os.getcwd(), "Results")
    if os.path.exists(src_dir):
        for fn in os.listdir(src_dir):
            if name in fn:
                s = os.path.join(src_dir, fn)
                d = os.path.join(MAPS_DIR, fn)
                if os.path.exists(d): os.remove(d)
                shutil.move(s, d)
    return {"variant_key": variant_key, "lattice": lattice, "mapshape": mapshape,
            "initialization": "pca", "total_epochs": TOTAL_EPOCHS,
            "train_rough_len": tp["train_rough_len"],
            "train_finetune_len": tp["train_finetune_len"],
            "train_rough_radiusin": tp["train_rough_radiusin"],
            "train_rough_radiusfin": RADIUS_FINAL,
            "quantization_error": float(qe), "topographic_error": float(te)}

def main():
    X, y = load_synthetic_control()
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    results = [train_variant(X, l, m, k) for l, m, k in VARIANTS]
    out = os.path.join(METRICS_DIR, "som_variants_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved " + out)

if __name__ == "__main__":
    main()
