<div align="center">

  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=120&section=header"/>

  <h1>
    <img src="https://readme-typing-svg.herokuapp.com/?lines=📰+CLUSTERIZAÇÃO+DE+NOTÍCIAS+COM+LLM;Clusterização+de+Texto+Não-Supervisionada;Rotulagem+Semântica+via+LLM;Dashboard+Visual+Interativo&font=Fira+Code&center=true&width=700&height=50&color=7aa2f7&vCenter=true&pause=1000&size=26" />
  </h1>

  <samp>Agrupar · Comparar · Explicar · Visualizar</samp>
  <br/><br/>

  <img src="https://img.shields.io/badge/Python-3.10+-c0caf5?style=for-the-badge&logo=python&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/React-18-7aa2f7?style=for-the-badge&logo=react&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML_Engine-9ece6a?style=for-the-badge&logo=scikitlearn&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/Gemini-LLM_Labels-bb9af7?style=for-the-badge&logo=google&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-f7768e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Concluído-9ece6a?style=for-the-badge"/>

</div>

<br/>

> Um pipeline de pesquisa acadêmica ponta a ponta que aplica **clusterização não-supervisionada** a um corpus de **315 notícias em português**, comparando múltiplas representações textuais e algoritmos de agrupamento. Os clusters são rotulados automaticamente por **LLMs (Gemini + Ollama)** e explorados através de um **dashboard interativo com tema Tokyo Night**, com projeções em tempo real e busca semântica.

<br/>

## `> stack_tecnológica`

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,react,typescript,vite,git&theme=dark&perline=7" />
</div>

<table align="center">
<tr>
<td align="center" width="33%">

**⚙️ Pipeline de ML**<br/><br/>
<img src="https://img.shields.io/badge/scikit--learn-1.3+-c0caf5?style=flat-square&logo=scikitlearn&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/SentenceTransformers-Multilingual-7aa2f7?style=flat-square"/>
<img src="https://img.shields.io/badge/UMAP-Redução_de_Dimensionalidade-9ece6a?style=flat-square"/>

</td>
<td align="center" width="33%">

**🧠 Rotulagem via LLM**<br/><br/>
<img src="https://img.shields.io/badge/Gemini-gemma--4--31b--it-bb9af7?style=flat-square&logo=google&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Ollama-Inferência_Local-f7768e?style=flat-square"/>
<img src="https://img.shields.io/badge/JSON_Mode-Saída_Estruturada-c0caf5?style=flat-square"/>

</td>
<td align="center" width="33%">

**📊 Dashboard**<br/><br/>
<img src="https://img.shields.io/badge/React_18-TypeScript-7aa2f7?style=flat-square&logo=react&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Vite-Canvas_2D-9ece6a?style=flat-square&logo=vite&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Zustand-Gerenciamento_de_Estado-bb9af7?style=flat-square"/>

</td>
</tr>
</table>

<br/>

## `> visão_geral_da_arquitetura`

```
news-clustering-llm/
│
├── 🐍 backend/
│   ├── pipeline.py              # Orquestração completa ML + LLM (~666 linhas)
│   ├── requirements.txt         # Dependências Python
│   ├── .env                     # Chaves de API (Gemini / host Ollama)
│   └── outputs/
│       ├── clustering_results.json   # Resultados completos pré-computados para o frontend
│       └── comparacao_agrupamento_*.png  # Gráficos comparativos estáticos (PCA/UMAP/t-SNE)
│
├── 🌐 frontend/
│   └── src/
│       ├── components/
│       │   ├── ClusterPlot.tsx       # Scatter plot Canvas 2D (zoom, pan, morphing)
│       │   ├── ClusterDetails.tsx    # Sidebar: rótulos LLM, estatísticas, leitor de documentos
│       │   ├── MetricsPanel.tsx      # Tabela comparativa de métricas (ARI, NMI, Purity)
│       │   └── LLMComparison.tsx     # Comparação lado a lado Gemini vs Ollama
│       └── store/
│           └── useClusteringStore.ts # Estado global Zustand
│
├── 📓 notebook_experimentos.ipynb   # Análise exploratória em Jupyter
└── 📄 Base_dados_textos_6_classes.csv  # Dataset bruto (315 artigos, 6 categorias)
```

<br/>

## `> fluxo_do_pipeline`

<table align="center">
<tr>
<td width="50%">

### 📥 Fase 1: Ingestão de Dados
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**Entrada:** `Base_dados_textos_6_classes.csv`
- **315** artigos em **6 categorias** de notícias
- Colunas: `Texto Original`, `Texto Expandido`, `Classe`, `Categoria`
- Filtragem de stopwords em PT · limpeza de valores nulos

</td>
<td width="50%">

### 🔢 Fase 2: Representações Textuais
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**3 representações comparadas:**
1. **TF-IDF** — top-1000 features, stopwords em PT
2. **Sentence Transformers** — `paraphrase-multilingual-MiniLM-L12-v2` (otimizado PT-BR)
3. **Ollama** — `nomic-embed-text` (vetores locais de 768 dimensões)

</td>
</tr>
<tr>
<td width="50%">

### 🤖 Fase 3: Algoritmos de Clusterização
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**4 algoritmos × 3 representações = 12 configurações:**
- **K-Means** — k=6, n_init=10
- **Agglomerative** — linkage Ward
- **DBSCAN** — ε adaptativo, min_samples=3
- **HDBSCAN** — min_cluster_size=5

</td>
<td width="50%">

### 🧠 Fase 4: Rotulagem Semântica via LLM
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**Estratégia dupla de LLM:**
- **Gemini** (`gemma-4-31b-it`) — Nuvem, saída JSON estruturada
- **Ollama** — Fallback local, privacidade total
- Saída: rótulo, resumo, análise de fronteira, coerência

</td>
</tr>
<tr>
<td width="50%">

### 📐 Fase 5: Redução de Dimensionalidade
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**3 projeções por configuração:**
- **PCA** — baseline linear
- **t-SNE** — preserva vizinhança (perplexidade auto-ajustada)
- **UMAP** — preserva topologia (n_neighbors=15)

</td>
<td width="50%">

### 📊 Fase 6: Dashboard Interativo
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**Visualização React + Canvas 2D:**
- Animação de transição suave (400ms ease-in-out) entre projeções
- Busca de documentos em tempo real com destaque de texto
- Efeito de pulso na seleção · Zoom/pan com eventos nativos

</td>
</tr>
</table>

<br/>

## `> resultados_e_benchmark`

<div align="center">
<img src="https://img.shields.io/badge/Dataset-315_Artigos-f7768e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Categorias-6_Classes-7aa2f7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Configurações-12_Combinações-bb9af7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Melhor_Purity-93.3%25-9ece6a?style=for-the-badge"/>
</div>

<br/>

**Tabela completa de métricas (ARI · NMI · Purity · Silhouette):**

| Configuração | ARI ↑ | NMI ↑ | Purity ↑ | Silhouette ↑ |
|:---|:---:|:---:|:---:|:---:|
| 🏆 **TF-IDF + Agglomerative** | **0.852** | **0.852** | **0.933** | 0.025 |
| TF-IDF + K-Means | 0.705 | 0.778 | 0.857 | 0.024 |
| ST + Agglomerative | 0.587 | 0.743 | 0.756 | 0.079 |
| ST + K-Means | 0.556 | 0.665 | 0.752 | 0.083 |
| Ollama + K-Means | 0.500 | 0.616 | 0.724 | 0.064 |
| Ollama + Agglomerative | 0.455 | 0.564 | 0.695 | 0.056 |
| TF-IDF + DBSCAN | 0.029 | 0.369 | 0.994* | 0.007 |
| ST + HDBSCAN | 0.091 | 0.221 | 0.429 | 0.173 |

> \* A purity do DBSCAN é inflada por colapsar a maioria dos artigos em clusters de ruído (-1).

**Principais descobertas:**
- **TF-IDF + Agglomerative (Ward)** é o vencedor geral — ARI/NMI altos indicam forte alinhamento com as categorias reais (ground-truth)
- **Sentence Transformers** (modelo multilingual PT-BR) tem desempenho inferior ao TF-IDF neste dataset — provavelmente por conta do texto curto, estilo manchete, do `Texto Original`; a representação de texto expandido reduz parcialmente essa diferença
- **DBSCAN / HDBSCAN** têm dificuldade nesses embeddings de alta dimensionalidade e distribuição uniforme — métodos baseados em densidade exigem estrutura de região densa que vetores de frequência/semânticos não formam naturalmente aqui

<br/>

## `> funcionalidades_do_dashboard`

<table align="center">
<tr>
<td width="50%">

**🗺️ Cluster Plot (Canvas 2D)**
- Alterna entre **12 combinações** de algoritmo/representação
- Alterna projeções: **PCA · UMAP · t-SNE**
- **Animação de transição** suave entre projeções
- Zoom (scroll) + pan (arraste) com eventos nativos
- Colore por **cluster previsto** ou **categoria real**
- Legenda com toggle individual por cluster

</td>
<td width="50%">

**📖 Sidebar de Detalhes do Cluster**
- **Rótulo semântico via LLM** com badge de coerência
- **Barras de pureza por categoria** — distribuição da classe real dentro do cluster
- **Documentos medoides** (mais próximos do centroide) vs **documentos de fronteira** (menor silhouette)
- **Leitor de documentos** — texto completo + texto expandido + rótulo previsto vs real
- **Busca em tempo real** com destaque de correspondências entre os 315 artigos

</td>
</tr>
<tr>
<td width="50%">

**📈 Painel de Métricas**
- Tabela comparativa completa das 12 configurações
- Destaca automaticamente a melhor configuração ao carregar

</td>
<td width="50%">

**🤖 Comparação de LLMs**
- Comparação lado a lado **Gemini vs Ollama** por cluster
- Saída estruturada em JSON: rótulo · resumo · análise de fronteira · coerência

</td>
</tr>
</table>

<br/>

## `> categorias_de_notícias`

O dataset cobre **6 classes temáticas** de notícias em português brasileiro:

| # | Categoria | Descrição |
|:---:|:---|:---|
| 1 | 🔵 **Economia** | Notícias econômicas, mercados, finanças |
| 2 | 🟠 **Esportes** | Cobertura esportiva, campeonatos |
| 3 | 🔴 **Polícia e Direitos** | Ocorrências policiais, direitos civis, justiça |
| 4 | 🟣 **Política** | Governo, eleições, políticas públicas |
| 5 | 🩵 **Turismo** | Viagens, cultura, turismo |
| 6 | 🟡 **Variedades e Sociedade** | Estilo de vida, sociedade, entretenimento |

<br/>

## `> instalação`

```bash
# 1. Clonar o repositório
git clone https://github.com/takaokensei/news-clustering-llm.git
cd news-clustering-llm
```

### 🐍 Backend (Pipeline de ML)

```bash
cd backend

# Instalar dependências Python
pip install -r requirements.txt

# Configurar chaves de API
cp .env.example .env
# Editar .env e definir: GEMINI_API_KEY e, opcionalmente, OLLAMA_HOST

# (Opcional) Subir o Ollama para inferência local
ollama serve
ollama pull llama3
ollama pull nomic-embed-text

# Rodar o pipeline completo (~5-15 min dependendo do hardware)
python pipeline.py
```

> O pipeline gera `backend/outputs/clustering_results.json` — a única fonte de verdade consumida pelo frontend.

### 🌐 Frontend (Dashboard)

```bash
cd frontend

# Instalar dependências Node.js
npm install

# Iniciar o servidor de desenvolvimento
npm run dev
# → http://localhost:5173
```

<br/>

## `> schema_de_saída_do_llm`

Cada cluster recebe uma explicação estruturada em JSON de ambos os backends de LLM:

```json
{
  "rotulo": "Economia & Mercados",
  "resumo": "Este cluster agrupa notícias sobre movimentações no mercado financeiro...",
  "analise_fronteira": "Os textos de fronteira misturam temas de política econômica...",
  "coerencia": "Alta"
}
```

Campos: **`rotulo`** (rótulo curto, ≤4 palavras) · **`resumo`** (resumo de 3–4 linhas) · **`analise_fronteira`** (ambiguidade de fronteira) · **`coerencia`** (`Alta / Média / Baixa`)

<br/>

## `> contato`

<div align="center">

  <strong>Cauã Vitor (takaokensei)</strong>
  <br/>
  <samp>Pesquisador de IA & Estudante de Engenharia Elétrica</samp>
  <br/>
  <samp>UFRN — Universidade Federal do Rio Grande do Norte</samp>

  <br/><br/>

  <a href="https://github.com/takaokensei">
    <img src="https://img.shields.io/badge/-GitHub-1a1b26?style=for-the-badge&logo=github&logoColor=c0caf5"/>
  </a>

</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Feito_com-Python_🐍_+_React_⚛️-c0caf5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Powered_by-Gemini_+_Ollama-7aa2f7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tema-Tokyo_Night_🌃-bb9af7?style=for-the-badge"/>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=100&section=footer"/>
