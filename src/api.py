import os
import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IntraSOM Local Inference API")

# Enable CORS so the Vercel frontend can call this localhost API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
PUBLIC_DATA_DIR = os.path.join(BASE_DIR, "frontend", "public", "data")

# Global variables for models
sbert_model = None

# Text metadata loaded from text_models.json (has_variants structure after v9)
text_metadata = None

@app.on_event("startup")
def startup_event():
    global sbert_model, text_metadata
    
    print("Loading models and metadata...")
    
    # 1. Load SBERT sentence transformer
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence-BERT model loaded successfully.")
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        
    # 2. Load metadata from text_models.json (now has has_variants structure)
    try:
        with open(os.path.join(PUBLIC_DATA_DIR, "text_models.json"), "r", encoding="utf-8") as f:
            text_metadata = json.load(f)
        print("Text models metadata loaded.")
    except Exception as e:
        print(f"Error loading text_models.json metadata: {e}")

class QueryRequest(BaseModel):
    text: str
    representation: str  # 'SBERT' or 'TF-IDF'
    dataset: str = '20news'   # '20news' or '6class'
    lattice: str = 'HEX'     # 'HEX' or 'RECT' — selects model variant

@app.get("/health")
def health():
    return {"status": "ok", "backend": "Python/FastAPI"}

@app.post("/classify")
def classify(req: QueryRequest):
    global sbert_model, text_metadata
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    rep = req.representation.upper()
    if rep not in ['SBERT', 'TF-IDF', 'TF_IDF']:
        raise HTTPException(status_code=400, detail="Invalid representation mode")
        
    if rep == 'TF_IDF':
        rep = 'TF-IDF'
        
    dname = req.dataset
    if dname not in ['20news', '6class']:
        dname = '20news'

    use_rect = req.lattice.upper() == 'RECT'
        
    try:
        # --- 1. Compute 20D embedding vector vec_20 ---
        if rep == 'SBERT':
            if sbert_model is None:
                raise HTTPException(status_code=500, detail="SBERT model not loaded")
                
            with open(os.path.join(MAPS_DIR, f"{dname}_sbert_pca.pkl"), "rb") as f:
                pca = pickle.load(f)
                
            emb = sbert_model.encode([req.text])[0]
            vec_20 = pca.transform([emb])[0]

            # Apply RECT Z-score scaler if lattice='RECT'
            if use_rect:
                scaler_path = os.path.join(MAPS_DIR, f"{dname}_SBERT_rect_scaler.pkl")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                vec_20 = scaler.transform([vec_20])[0]
        else:
            with open(os.path.join(MAPS_DIR, f"{dname}_tfidf_vectorizer.pkl"), "rb") as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(MAPS_DIR, f"{dname}_lsa_svd.pkl"), "rb") as f:
                svd = pickle.load(f)
                
            tfidf_vec = vectorizer.transform([req.text])
            vec_20 = svd.transform(tfidf_vec)[0]

            # Apply RECT Z-score scaler if lattice='RECT'
            if use_rect:
                scaler_path = os.path.join(MAPS_DIR, f"{dname}_TF-IDF_rect_scaler.pkl")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                vec_20 = scaler.transform([vec_20])[0]

        # --- 2. Load codebook for BMU distance search ---
        if use_rect:
            # RECT_planar: codebooks from text_rect_models.json
            rect_path = os.path.join(MAPS_DIR, "text_rect_models.json")
            with open(rect_path, "r", encoding="utf-8") as f:
                rect_data = json.load(f)
            neurons_json = rect_data.get(dname, {}).get(rep, {}).get("RECT_planar", {}).get("neurons", [])
            if not neurons_json:
                raise HTTPException(status_code=404, detail=f"RECT model for {dname}/{rep} not found")
            codebooks = np.array([n["codebook"] for n in neurons_json])   # shape (100, 20)
            bmu_ids   = [n["id"] for n in neurons_json]
        else:
            # HEX_toroid: codebooks from pre-trained .parquet (original path)
            neurons_file = os.path.join(MAPS_DIR, f"SOM_Text_{dname}_{rep}_neurons.parquet")
            if not os.path.exists(neurons_file):
                raise HTTPException(status_code=404, detail=f"HEX model for {dname}/{rep} not found")
            neurons_df = pd.read_parquet(neurons_file)
            dim_cols   = [f"B_Dim_{i}" for i in range(1, 21)]
            codebooks  = neurons_df[dim_cols].values   # shape (100, 20)
            bmu_ids    = neurons_df["BMU"].tolist()

        # --- 3. BMU search via Euclidean distance ---
        diffs = codebooks - vec_20
        dists = np.linalg.norm(diffs, axis=1)   # shape (100,)
        min_idx  = int(np.argmin(dists))
        bmu_id   = bmu_ids[min_idx]
        bmu_dist = float(dists[min_idx])
        
        # --- 4. Metadata (dominant class, purity) from text_models.json ---
        dominant_class = "Desconhecido"
        purity = 0.0
        
        # Navigate has_variants structure introduced in export_data_for_frontend.py v9
        rep_entry = text_metadata.get(dname, {}).get(rep, {}) if text_metadata else {}
        if "has_variants" in rep_entry:
            variant_key = "RECT_planar" if use_rect else "HEX_toroid"
            meta_neurons = rep_entry.get(variant_key, {}).get("neurons", [])
        else:
            # Backward-compatible: pre-v9 flat structure
            meta_neurons = rep_entry.get("neurons", [])

        meta_neuron = next((n for n in meta_neurons if n["id"] == bmu_id), None)
        if meta_neuron:
            dominant_class = meta_neuron["dominant_class"]
            purity = meta_neuron["purity"]
            
        # --- 5. Confidence score ---
        avg_dist = np.mean(dists)
        confidence = max(0, min(100, int((1.0 - (bmu_dist / (avg_dist or 1.0))) * 100)))
        
        # ESPELHADO: Esta lógica de escala/ajuste de confiança está espelhada no frontend
        # (lib/pca.ts). Qualquer alteração nesta lógica de ajuste de pontuação deve ser
        # replicada manualmente nos dois arquivos.
        # NOTE: api.py usa avg_dist como denominador (escala relativa), enquanto pca.ts usa
        # maxExpectedDist=6.0 (escala absoluta, calibrada para o HEX). São abordagens
        # distintas — api.py não precisou de recalibração para RECT porque a escala relativa
        # (bmu_dist / avg_dist) se adapta automaticamente à escala da normalização.
        confidence = int(50 + (confidence / 2.0)) if confidence > 0 else 0
        if bmu_dist < 1.0:
            confidence = max(confidence, 95)
            
        return {
            "bmu": bmu_id,
            "dominantClass": dominant_class,
            "purity": purity,
            "score": confidence,
            "distance": bmu_dist
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
