import os
import shutil
from huggingface_hub import HfApi

TOKEN = "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"
REPO_ID = "takaokensei/intrasom-kohonen-api"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DIR = os.path.join(BASE_DIR, "hf_space_files")
DATA_DIR = os.path.join(HF_DIR, "data")

print("Initializing Hugging Face Space folder...")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Copy maps and metadata files to hf_space_files/data
maps_source = os.path.join(BASE_DIR, "outputs", "maps")
public_source = os.path.join(BASE_DIR, "frontend", "public", "data")

files_to_copy = [
    (os.path.join(maps_source, "sbert_pca.pkl"), os.path.join(DATA_DIR, "sbert_pca.pkl")),
    (os.path.join(maps_source, "tfidf_vectorizer.pkl"), os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")),
    (os.path.join(maps_source, "lsa_svd.pkl"), os.path.join(DATA_DIR, "lsa_svd.pkl")),
    (os.path.join(maps_source, "SOM_Text_SBERT_neurons.parquet"), os.path.join(DATA_DIR, "SOM_Text_SBERT_neurons.parquet")),
    (os.path.join(maps_source, "SOM_Text_TF-IDF_neurons.parquet"), os.path.join(DATA_DIR, "SOM_Text_TF-IDF_neurons.parquet")),
    (os.path.join(public_source, "text_models.json"), os.path.join(DATA_DIR, "text_models.json")),
]

for src, dest in files_to_copy:
    if os.path.exists(src):
        shutil.copy2(src, dest)
        print(f"Copied {os.path.basename(src)} to Hugging Face data folder.")
    else:
        print(f"Warning: {src} not found!")

# 2. Write Dockerfile
dockerfile_content = """FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""
with open(os.path.join(HF_DIR, "Dockerfile"), "w") as f:
    f.write(dockerfile_content)
print("Created Dockerfile.")

# 3. Write requirements.txt
requirements_content = """fastapi
uvicorn
sentence-transformers
scikit-learn
pandas
pyarrow
pydantic
"""
with open(os.path.join(HF_DIR, "requirements.txt"), "w") as f:
    f.write(requirements_content)
print("Created requirements.txt.")

# 4. Write app.py
app_content = """import os
import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IntraSOM Cloud Inference API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

sbert_model = None
sbert_pca = None
tfidf_vectorizer = None
lsa_svd = None
sbert_neurons_df = None
tfidf_neurons_df = None
text_metadata = None

@app.on_event("startup")
def startup_event():
    global sbert_model, sbert_pca, tfidf_vectorizer, lsa_svd
    global sbert_neurons_df, tfidf_neurons_df, text_metadata
    
    print("Loading models and metadata on Hugging Face startup...")
    
    # 1. Load SBERT sentence transformer
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SBERT model loaded.")
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        
    # 2. Load Pickle files
    try:
        with open(os.path.join(DATA_DIR, "sbert_pca.pkl"), "rb") as f:
            sbert_pca = pickle.load(f)
        with open(os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(DATA_DIR, "lsa_svd.pkl"), "rb") as f:
            lsa_svd = pickle.load(f)
        print("Pickle models loaded.")
    except Exception as e:
        print(f"Error loading pickle models: {e}")
        
    # 3. Load Parquet codebooks
    try:
        sbert_neurons_df = pd.read_parquet(os.path.join(DATA_DIR, "SOM_Text_SBERT_neurons.parquet"))
        tfidf_neurons_df = pd.read_parquet(os.path.join(DATA_DIR, "SOM_Text_TF-IDF_neurons.parquet"))
        print("Neurons parquet files loaded.")
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        
    # 4. Load metadata from text_models.json
    try:
        with open(os.path.join(DATA_DIR, "text_models.json"), "r", encoding="utf-8") as f:
            text_metadata = json.load(f)
        print("Metadata loaded.")
    except Exception as e:
        print(f"Error loading text_models.json: {e}")

class QueryRequest(BaseModel):
    text: str
    representation: str

@app.get("/")
def home():
    return {"message": "IntraSOM Cloud Inference API is running on Hugging Face Spaces!", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "backend": "Hugging Face Space"}

@app.post("/classify")
def classify(req: QueryRequest):
    global sbert_model, sbert_pca, tfidf_vectorizer, lsa_svd
    global sbert_neurons_df, tfidf_neurons_df, text_metadata
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    rep = req.representation.upper()
    if rep not in ['SBERT', 'TF-IDF', 'TF_IDF']:
        raise HTTPException(status_code=400, detail="Invalid representation mode")
        
    if rep == 'TF_IDF':
        rep = 'TF-IDF'
        
    try:
        if rep == 'SBERT':
            if sbert_model is None or sbert_pca is None or sbert_neurons_df is None:
                raise HTTPException(status_code=500, detail="SBERT models not initialized")
            emb = sbert_model.encode([req.text])[0]
            vec_20 = sbert_pca.transform([emb])[0]
            neurons_df = sbert_neurons_df
        else:
            if tfidf_vectorizer is None or lsa_svd is None or tfidf_neurons_df is None:
                raise HTTPException(status_code=500, detail="TF-IDF models not initialized")
            tfidf_vec = tfidf_vectorizer.transform([req.text])
            vec_20 = lsa_svd.transform(tfidf_vec)[0]
            neurons_df = tfidf_neurons_df
            
        dim_cols = [f"B_Dim_{i}" for i in range(1, 21)]
        codebooks = neurons_df[dim_cols].values
        
        diffs = codebooks - vec_20
        dists = np.linalg.norm(diffs, axis=1)
        
        min_idx = np.argmin(dists)
        bmu_row = neurons_df.iloc[min_idx]
        bmu_id = int(bmu_row['BMU'])
        bmu_dist = float(dists[min_idx])
        
        dominant_class = "Desconhecido"
        purity = 0.0
        
        meta_neurons = text_metadata.get(rep, {}).get("neurons", [])
        meta_neuron = next((n for n in meta_neurons if n["id"] == bmu_id), None)
        if meta_neuron:
            dominant_class = meta_neuron["dominant_class"]
            purity = meta_neuron["purity"]
            
        avg_dist = np.mean(dists)
        confidence = max(0, min(100, int((1.0 - (bmu_dist / (avg_dist or 1.0))) * 100)))
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
        raise HTTPException(status_code=500, detail=str(e))
"""
with open(os.path.join(HF_DIR, "app.py"), "w") as f:
    f.write(app_content)
print("Created app.py.")

# 5. Push to Hugging Face Spaces
print("Authenticating and pushing to Hugging Face Spaces...")
try:
    api = HfApi(token=TOKEN)
    
    # Try to create repository, ignore if it already exists
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            private=False
        )
        print(f"Created Hugging Face Space repository: {REPO_ID}")
    except Exception as e:
        print("Repository already exists or could not be created. Proceeding to upload folder...")
        
    # Upload folder
    print(f"Uploading folder to HF Space: {REPO_ID} ...")
    api.upload_folder(
        folder_path=HF_DIR,
        repo_id=REPO_ID,
        repo_type="space"
    )
    print("Hugging Face Space successfully deployed!")
    print(f"Space URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"API URL: https://takaokensei-intrasom-kohonen-api.hf.space")
except Exception as e:
    print(f"Error during Hugging Face deployment: {e}")
