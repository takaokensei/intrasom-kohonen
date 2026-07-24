<div align="center">

  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=120&section=header"/>

  <h1>
    <img src="https://readme-typing-svg.herokuapp.com/?lines=🧠+INTRASOM+KOHONEN+MAPS+ANALYZER;Self-Organizing+Maps+(SOM);Time+Series+%26+NLP;Interactive+Dashboard&font=Fira+Code&center=true&width=700&height=50&color=7aa2f7&vCenter=true&pause=1000&size=26" />
  </h1>

  <samp>Interactive Visualization · Self-Organizing Maps (SOM) · Time Series · Natural Language Processing</samp>
  <br/><br/>

  <img src="https://img.shields.io/badge/React-19-7aa2f7?style=for-the-badge&logo=react&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/TypeScript-Strict_Mode-3178c6?style=for-the-badge&logo=typescript&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Vite-Built_System-646cff?style=for-the-badge&logo=vite&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Status-Completed-9ece6a?style=for-the-badge"/>

</div>

<br/>

> A high-performance interactive academic dashboard designed to analyze and explore **Kohonen Self-Organizing Maps (SOM)** applied to two distinct fronts: **synthetic time series** and **semantic text clustering**.
>
> Built with a premium aesthetic based on the **Tokyo Night** theme, the system features full visualization controls, smooth animations, and a hybrid AI inference pipeline (with local execution or 100% serverless cloud fallback).

<div align="center">

  <a href="https://intrasom-kohonen.vercel.app"><img src="https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white"/></a>

</div>

<br/>

## `> tech_stack`

<div align="center">
  <img src="https://skillicons.dev/icons?i=react,typescript,vite,python,git&theme=dark&perline=7" />
</div>

<table align="center">
<tr>
<td align="center" width="33%">

**🌐 Frontend**<br/><br/>
<img src="https://img.shields.io/badge/React_19-TypeScript-7aa2f7?style=flat-square&logo=react&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Vite-Build_System-9ece6a?style=flat-square&logo=vite&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Zustand-State_Management-bb9af7?style=flat-square"/>

</td>
<td align="center" width="33%">

**🐍 Backend / AI**<br/><br/>
<img src="https://img.shields.io/badge/FastAPI-Local_Inference-c0caf5?style=flat-square&logo=fastapi&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Sentence--BERT-Semantic_Embeddings-f7768e?style=flat-square"/>
<img src="https://img.shields.io/badge/TF--IDF-Frequency_Representation-7aa2f7?style=flat-square"/>

</td>
<td align="center" width="33%">

**🧩 Data**<br/><br/>
<img src="https://img.shields.io/badge/Synthetic_Control-600_Series-9ece6a?style=flat-square"/>
<img src="https://img.shields.io/badge/20_Newsgroups-400_News-bb9af7?style=flat-square"/>
<img src="https://img.shields.io/badge/HuggingFace-Inference_API-c0caf5?style=flat-square&logo=huggingface&logoColor=1a1b26"/>

</td>
</tr>
</table>

<br/>

## `> system_overview`

The dashboard is structured around two main areas of analysis:

<table align="center">
<tr>
<td width="50%">

### 📈 1. Time Series (Synthetic Control)
<p><img src="https://img.shields.io/badge/Status-✅_Complete-9ece6a?style=for-the-badge"/></p>

**Dataset:** 600 time series from the *Synthetic Control* dataset, split across 6 trend classes (Normal, Cyclic, Increasing Trend, Decreasing Trend, Upward Shift, and Downward Shift).

**Supported maps:** Hexagonal grids of sizes **5x5, 7x7, 10x10, 12x12, 15x15, and 20x20**.

**Visualizations:**
- **U-Matrix** — distances between neighboring neurons on the lattice
- **Dominant Class** — majority trend class per neuron
- **Purity** — internal cohesion of each neuron
- **Baseline Metrics** — real-time comparison (ARI, NMI, Silhouette, Davies-Bouldin, Calinski-Harabasz, Quantization Error, Topographic Error) against K-Means, PCA + K-Means, Agglomerative Clustering, and DBSCAN
- **Radar Chart** — trade-offs across SOM map dimensions

</td>
<td width="50%">

### 📰 2. Semantic Text Classifier (20 Newsgroups)
<p><img src="https://img.shields.io/badge/Status-✅_Complete-9ece6a?style=for-the-badge"/></p>

**Dataset:** 400 news articles from the *20 Newsgroups* corpus, evenly distributed across 4 target classes (Graphics, Baseball, Mideast, and Space).

**Hexagonal maps (10x10):**
- Side-by-side comparison between the classic **TF-IDF (Frequency-based)** representation and the deep **Sentence-BERT (Semantic)** representation
- Statistical analysis highlighting the ARI/NMI gain achieved with contextual embeddings

**Real-time classifier:**
- Type any short or long text and project it onto the Kohonen lattice to find the corresponding activated neuron (BMU)
- **Cascading hybrid pipeline** with three fallback levels:
  1. Tries the **local FastAPI server** (`http://127.0.0.1:8000`), when available
  2. If offline, calls the **Hugging Face Inference API** (via a serverless proxy function) and runs the 384D → 20D PCA projection in JavaScript directly in the browser
  3. If the external call fails, falls back to a **client-side heuristic** based on keyword similarity, ensuring the classifier is never fully unavailable

</td>
</tr>
</table>

<br/>

## `> file_structure`

```
intrasom-kohonen/
│
├── 🌐 frontend/                      # React + TS web application (deployed on Vercel)
│   ├── 📁 api/                       # Serverless function (proxy to the HF Inference API)
│   ├── 📁 public/data/               # Pre-computed static data (CDN)
│   │   ├── series.json               # 600 time series
│   │   ├── som_models.json           # 5x5 to 20x20 lattices for the time series maps
│   │   ├── text_models.json          # Text lattices (SBERT + TF-IDF)
│   │   └── pca_params.json           # PCA mean and components (384D -> 20D)
│   └── 📁 src/
│       ├── 📁 components/            # HexGrid, TextHexGrid, ClassifierPanel, etc.
│       └── 📁 store/
│           └── useDashboardStore.ts  # Global dashboard state and inference logic
│
├── 🐍 src/                           # Python scripts and AI pipeline
│   ├── api.py                        # Local FastAPI server for vectorization/projection
│   ├── train_som.py                  # Training of the base models (HEX toroidal)
│   ├── train_som_variants.py         # Planar variants (HEX)
│   ├── train_som_rect.py             # Rectangular variant via MiniSom
│   ├── train_parameter_study.py      # Multivariate parameter sensitivity study
│   ├── text_som_clustering.py        # Training pipeline for the textual SOM
│   ├── export_data_for_frontend.py   # Exports Python results into static JSONs
│   ├── evaluate_clusters.py          # Evaluation metrics (ARI, NMI, Silhouette, etc.)
│   └── deploy_to_hf.py               # Packages and publishes the API to Hugging Face Spaces
│
├── 🚀 hf_space_files/                # Mirror of the inference API hosted on Hugging Face Spaces
├── 📓 notebooks/                     # EDA, training, and visualizations (Jupyter)
├── 📚 docs/                          # Supplementary technical documentation
└── ✅ tests/                         # Automated tests (pytest)
```

<br/>

## `> running_and_development`

**Prerequisites:**
<img src="https://img.shields.io/badge/Node.js-v20+-9ece6a?style=flat-square&logo=node.js&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Python-v3.10+-3776ab?style=flat-square&logo=python&logoColor=ffffff"/>

### 🌐 Frontend (React + Vite)

```bash
# 1. Navigate to the frontend folder
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
# → http://localhost:5173
```

### 🐍 Local Backend API (Optional)

To enable real-time inference with native TF-IDF or to process embeddings locally at higher speed:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server from the project root
python src/api.py
# → http://127.0.0.1:8000
```

> Once the API is running, the frontend automatically switches its status indicator to **Local Server: ACTIVE**.

### ☁️ Inference API Deployment (Hugging Face Spaces)

The same API can also be published as a Hugging Face Space (`hf_space_files/`), serving as the cloud layer for the hybrid pipeline whenever no local backend is available:

```bash
# Package the trained artifacts and publish to Hugging Face Spaces
python src/deploy_to_hf.py
```

<br/>

## `> retraining_the_models`

To reproduce training with the project's core SOM parameters (PCA initialization, 500 epochs, 80%→1 neighborhood radius decay) or rebuild the frontend's static JSON files:

```bash
# 1. Make sure the Python dependencies are installed
pip install -r requirements.txt

# 2. Train all base Synthetic Control models (6 map sizes: HEX_toroid)
python src/train_som.py

# 3. Train the planar variants for ALL 6 sizes (HEX_planar / Toroid OFF)
python src/train_som_variants.py

# 4. Train the rectangular variants for ALL 6 sizes via MiniSom (RECT_planar)
python src/train_som_rect.py

# 5. Run the multivariate parameter sensitivity study
python src/train_parameter_study.py

# 6. Train the 4 semantic text models (20news and 6class, TF-IDF and SBERT)
python src/text_som_clustering.py

# 7. Export all structured files (.parquet + JSONs) for the React frontend
python src/export_data_for_frontend.py
```

<br/>

## `> engine_architecture`

* **Hexagonal Grid (`IntraSOM`):**
  All hexagonal maps (both toroidal and planar topologies) across the 6 map sizes (5x5 to 20x20) are trained with the primary `intrasom` engine.
* **Rectangular Grid (`MiniSom`):**
  The use of the complementary **MiniSom** engine (`minisom==2.3.6`) for the **RECT_planar** variant stems from a real technical limitation in the `intrasom` library:
  1. **Main Blocker (Unresolved):** The `build_umatrix()` method explicitly raises `Exception("build_umatrix error: non hexagonal lattice not implemented!")`, which would require rewriting the U-Matrix construction for rectangular lattices inside the library (out of scope).
  2. **Syntax Bug (Patched Locally):** The bug in `Codebook._rect_dist_plan` (a generator passed to `np.array()`) was fixed via a monkey-patch in `src/reproducibility.py` (`_patch_intrasom_rect_dist_plan`), though this fix alone does not remove the U-Matrix blocker.
  For this structural reason (and not as an arbitrary stylistic choice), we use **MiniSom** with the synchronous batch algorithm `train_batch_offline`, ensuring 100% real data with no decorative fallback.

<br/>

## `> testing`

The backend has automated tests via `pytest`, covering reproducibility (global seed) and the FastAPI endpoints:

```bash
pytest tests/
```

<br/>

## `> accessibility`

<table align="center">
<tr>
<td width="50%">

**⌨️ Keyboard Support**
The entire hexagonal lattice is navigable using the `Tab` key. The focused neuron can be activated using `Enter` or `Space`.

</td>
<td width="50%">

**🔊 Screen Readers**
Full ARIA tags (`role="button"`, `aria-label`) describe the neuron's ID, associated sample count, dominant class, and purity for each hexagon.

</td>
</tr>
</table>

<br/>

## `> contact`

<div align="center">

  <strong>Cauã Vitor (takaokensei)</strong>
  <br/>
  <samp>AI Researcher & Electrical Engineering Student</samp>
  <br/>
  <samp>UFRN — Federal University of Rio Grande do Norte</samp>

  <br/><br/>

  <a href="https://github.com/takaokensei">
    <img src="https://img.shields.io/badge/-GitHub-1a1b26?style=for-the-badge&logo=github&logoColor=c0caf5"/>
  </a>

</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Made_with-React_⚛️_+_Python_🐍-c0caf5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Powered_by-IntraSOM_+_Sentence--BERT-7aa2f7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Theme-Tokyo_Night_🌃-bb9af7?style=for-the-badge"/>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=100&section=footer"/>
