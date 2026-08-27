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
    representation: str = 'SBERT'  # 'SBERT', 'TF-IDF', 'BGE-M3', 'Gemma-300M'
    dataset: str = '20news'        # '20news' or '6class'
    lattice: str = 'HEX'           # 'HEX' or 'RECT' — selects model variant


class SearchRequest(BaseModel):
    query: str
    representation: str = 'SBERT'  # 'SBERT', 'TF-IDF', 'BGE-M3', 'Gemma-300M'
    dataset: str = '20news'        # '20news' or '6class'
    lattice: str = 'HEX'           # 'HEX' or 'RECT'
    mode: str = 'global'           # 'global' or 'topological'
    top_k: int = 5
    radius: int = 1


@app.get("/health")
def health():
    return {"status": "ok", "backend": "Python/FastAPI"}


def _get_query_vector(text: str, rep: str, dname: str, use_rect: bool):
    global sbert_model
    from sentence_transformers import SentenceTransformer

    if rep == 'SBERT':
        if sbert_model is None:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(os.path.join(MAPS_DIR, f"{dname}_sbert_pca.pkl"), "rb") as f:
            pca = pickle.load(f)
        emb = sbert_model.encode([text])[0]
        vec_20 = pca.transform([emb])[0]
        if use_rect:
            scaler_path = os.path.join(MAPS_DIR, f"{dname}_SBERT_rect_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                vec_20 = scaler.transform([vec_20])[0]
    elif rep in ['BGE-M3', 'BGE_M3']:
        pca_path = os.path.join(MAPS_DIR, f"{dname}_bgem3_pca.pkl")
        if not os.path.exists(pca_path):
            raise HTTPException(status_code=404, detail="BGE-M3 PCA parameters not found")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        bge_model = SentenceTransformer("BAAI/bge-m3")
        emb = bge_model.encode([text])[0]
        vec_20 = pca.transform([emb])[0]
    elif rep in ['GEMMA-300M', 'GEMMA_300M']:
        pca_path = os.path.join(MAPS_DIR, f"{dname}_gemma300m_pca.pkl")
        if not os.path.exists(pca_path):
            raise HTTPException(status_code=404, detail="Gemma-300M PCA parameters not found")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        gemma_model = SentenceTransformer("google/embeddinggemma-300m")
        emb = gemma_model.encode([text])[0]
        vec_20 = pca.transform([emb])[0]
    else:  # TF-IDF
        with open(os.path.join(MAPS_DIR, f"{dname}_tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MAPS_DIR, f"{dname}_lsa_svd.pkl"), "rb") as f:
            svd = pickle.load(f)
        tfidf_vec = vectorizer.transform([text])
        vec_20 = svd.transform(tfidf_vec)[0]
        if use_rect:
            scaler_path = os.path.join(MAPS_DIR, f"{dname}_TF-IDF_rect_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                vec_20 = scaler.transform([vec_20])[0]
    return vec_20


@app.post("/classify")
def classify(req: QueryRequest):
    global sbert_model, text_metadata
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    rep = req.representation.upper()
    if rep in ['TF_IDF', 'TFIDF']:
        rep = 'TF-IDF'
    elif rep == 'BGE_M3':
        rep = 'BGE-M3'
    elif rep in ['GEMMA_300M', 'GEMMA300M']:
        rep = 'Gemma-300M'
        
    dname = req.dataset if req.dataset in ['20news', '6class'] else '20news'
    use_rect = req.lattice.upper() == 'RECT'
        
    try:
        vec_20 = _get_query_vector(req.text, rep, dname, use_rect)

        # Load codebook for BMU distance search
        if use_rect:
            variant_key = "RECT_planar"
            neurons_file = os.path.join(MAPS_DIR, f"SOM_Text_{dname}_{rep}_RECT_planar_neurons.parquet")
        else:
            variant_key = "HEX_toroid"
            neurons_file = os.path.join(MAPS_DIR, f"SOM_Text_{dname}_{rep}_neurons.parquet")

        if not os.path.exists(neurons_file):
            raise HTTPException(status_code=404, detail=f"SOM model file {neurons_file} not found")

        neurons_df = pd.read_parquet(neurons_file)
        dim_cols = [f"B_Dim_{i}" for i in range(1, 21)]
        codebooks = neurons_df[dim_cols].values
        bmu_ids = neurons_df["BMU"].tolist()

        diffs = codebooks - vec_20
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = int(np.argmin(dists))
        bmu_id = bmu_ids[min_idx]
        bmu_dist = float(dists[min_idx])
        
        dominant_class = "Desconhecido"
        purity = 0.0
        entropy = 0.0
        
        rep_entry = text_metadata.get(dname, {}).get(rep, {}) if text_metadata else {}
        meta_neurons = rep_entry.get(variant_key, {}).get("neurons", []) if "has_variants" in rep_entry else rep_entry.get("neurons", [])
        meta_neuron = next((n for n in meta_neurons if n["id"] == bmu_id), None)
        if meta_neuron:
            dominant_class = meta_neuron.get("dominant_class", "Desconhecido")
            purity = meta_neuron.get("purity", 0.0)
            entropy = meta_neuron.get("entropy", 0.0)
            
        avg_dist = np.mean(dists)
        confidence = max(0, min(100, int((1.0 - (bmu_dist / (avg_dist or 1.0))) * 100)))
        confidence = int(50 + (confidence / 2.0)) if confidence > 0 else 0
        if bmu_dist < 1.0:
            confidence = max(confidence, 95)
            
        return {
            "bmu": bmu_id,
            "dominantClass": dominant_class,
            "purity": purity,
            "entropy": entropy,
            "score": confidence,
            "distance": bmu_dist
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    rep = req.representation.upper()
    if rep in ['TF_IDF', 'TFIDF']: rep = 'TF-IDF'
    elif rep == 'BGE_M3': rep = 'BGE-M3'
    elif rep in ['GEMMA_300M', 'GEMMA300M']: rep = 'Gemma-300M'

    dname = req.dataset if req.dataset in ['20news', '6class'] else '20news'
    use_rect = req.lattice.upper() == 'RECT'

    try:
        from src.search.semantic_search import search_semantic_global, search_topological_som

        # 1. Obter vetor da query (20D normalizado)
        query_vec = _get_query_vector(req.query, rep, dname, use_rect)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # 2. Carregar vetores dos documentos e results.parquet
        variant_suffix = "_RECT_planar" if use_rect else ""
        results_file = os.path.join(MAPS_DIR, f"SOM_Text_{dname}_{rep}{variant_suffix}_results.parquet")
        neurons_file = os.path.join(MAPS_DIR, f"SOM_Text_{dname}_{rep}{variant_suffix}_neurons.parquet")

        if not os.path.exists(results_file):
            raise HTTPException(status_code=404, detail=f"Results file {results_file} not found")

        results_df = pd.read_parquet(results_file)
        dim_cols = [f"B_Dim_{i}" for i in range(1, 21)]
        doc_embs = results_df[dim_cols].values
        # Normalizar documentos para similaridade de cosseno
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1.0
        doc_embs_norm = doc_embs / doc_norms

        # 3. Carregar textos das amostras se existirem
        samples_path = os.path.join(PUBLIC_DATA_DIR, "news_samples.json")
        sample_map = {}
        if os.path.exists(samples_path):
            with open(samples_path, "r", encoding="utf-8") as f:
                samples_data = json.load(f)
            ds_samples = samples_data.get(dname, [])
            for s in ds_samples:
                sample_map[s["id"]] = s

        # 4. Executar busca global ou topológica via BMU
        if req.mode.lower() == "topological" and os.path.exists(neurons_file):
            neurons_df = pd.read_parquet(neurons_file)
            codebooks = neurons_df[dim_cols].values
            cols = 10
            rows = 10
            ranked = search_topological_som(
                query_vec, codebooks, (cols, rows), results_df, doc_embs_norm, top_k=req.top_k, radius=req.radius
            )
        else:
            ranked = search_semantic_global(query_vec, doc_embs_norm, top_k=req.top_k)

        # 5. Formatar resultados
        results_list = []
        for doc_idx, score in ranked:
            bmu = int(results_df.iloc[doc_idx]["BMU"])
            sample_info = sample_map.get(doc_idx, {})
            text_preview = sample_info.get("text", f"Documento #{doc_idx + 1}")
            doc_class = sample_info.get("class", "Desconhecido")
            if len(text_preview) > 280:
                text_preview = text_preview[:280] + "..."

            results_list.append({
                "doc_id": doc_idx,
                "text": text_preview,
                "score": round(float(score), 4),
                "bmu": bmu,
                "class": doc_class
            })

        return {
            "query": req.query,
            "dataset": dname,
            "representation": rep,
            "mode": req.mode,
            "total_candidates": len(ranked),
            "results": results_list
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

