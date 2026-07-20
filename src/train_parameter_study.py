import os, sys, shutil, json, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control
from reproducibility import set_global_seed
from train_som import TOTAL_EPOCHS, RADIUS_FINAL
import intrasom

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(WORKSPACE_DIR, "outputs", "maps")
METRICS_DIR = os.path.join(WORKSPACE_DIR, "outputs", "metrics")

MAPSIZE = (10, 10)
MAX_SIDE = max(MAPSIZE)  # 10

# Study design: vary ONE parameter at a time, hold the rest at recommended values.
# Recommended: radiusin=8 (80%), radiusfin=1, rough=150, finetune=350, total=500.
STUDY_VARIANTS = [
    {"key": "recommended", "rough": 150, "finetune": 350, "rin": 8, "rfin": 1, "rfin_ft": 1},
    {"key": "initial_50",  "rough": 150, "finetune": 350, "rin": 5, "rfin": 1, "rfin_ft": 1},
    {"key": "initial_100", "rough": 150, "finetune": 350, "rin": 10,"rfin": 1, "rfin_ft": 1},
    {"key": "final_2",     "rough": 150, "finetune": 350, "rin": 8, "rfin": 2, "rfin_ft": 2},
    {"key": "epochs_100",  "rough": 30,  "finetune": 70,  "rin": 8, "rfin": 1, "rfin_ft": 1},
    {"key": "epochs_200",  "rough": 60,  "finetune": 140, "rin": 8, "rfin": 1, "rfin_ft": 1},
]

def train_variant(X, v):
    name = "SOM_study_" + v["key"]
    tp = {
        "train_rough_len":          v["rough"],
        "train_rough_radiusin":     v["rin"],
        "train_rough_radiusfin":    v["rfin"],
        "train_finetune_len":       v["finetune"],
        "train_finetune_radiusin":  v["rfin"],
        "train_finetune_radiusfin": v["rfin_ft"],
    }
    set_global_seed()
    som = intrasom.SOMFactory.build(data=X, mapsize=MAPSIZE, mapshape="toroid",
        lattice="hexa", normalization="var", initialization="pca",
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
    return {
        "key": v["key"],
        "total_epochs": v["rough"] + v["finetune"],
        "rough_epochs": v["rough"],
        "finetune_epochs": v["finetune"],
        "radius_initial": v["rin"],
        "radius_final": v["rfin_ft"],
        "radius_initial_pct": round(v["rin"] / MAX_SIDE * 100),
        "quantization_error": float(qe),
        "topographic_error": float(te),
        "label": {
            "recommended": "Recomendado (80%->1, 500ep)",
            "initial_50":  "Vizinhanca inicial 50% (->1, 500ep)",
            "initial_100": "Vizinhanca inicial 100% (->1, 500ep)",
            "final_2":     "Vizinhanca final 2 (80%->2, 500ep)",
            "epochs_100":  "100 Epocas (80%->1)",
            "epochs_200":  "200 Epocas (80%->1)",
        }.get(v["key"], v["key"]),
    }

def main():
    print("Loading data...")
    X, y = load_synthetic_control()
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    results = []
    for v in STUDY_VARIANTS:
        print("\nVariante:", v["key"])
        results.append(train_variant(X, v))
    out = os.path.join(METRICS_DIR, "parameter_study.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved", out)
    for r in results:
        print("  {:25s}  QE={:.4f}  TE={:.4f}".format(r["key"], r["quantization_error"], r["topographic_error"]))

if __name__ == "__main__":
    main()
