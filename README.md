<div align="center">

  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=120&section=header"/>

  <h1>
    <img src="https://readme-typing-svg.herokuapp.com/?lines=🧠+INTRASOM+KOHONEN+MAPS+ANALYZER;Mapas+Auto-Organizáveis+(SOM);Séries+Temporais+%26+NLP;Dashboard+Interativo&font=Fira+Code&center=true&width=700&height=50&color=7aa2f7&vCenter=true&pause=1000&size=26" />
  </h1>

  <samp>Visualização Interativa · Mapas Auto-Organizáveis (SOM) · Séries Temporais · Processamento de Linguagem Natural</samp>
  <br/><br/>

  <img src="https://img.shields.io/badge/React-19-7aa2f7?style=for-the-badge&logo=react&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/TypeScript-Strict_Mode-3178c6?style=for-the-badge&logo=typescript&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Vite-Built_System-646cff?style=for-the-badge&logo=vite&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Status-Concluído-9ece6a?style=for-the-badge"/>

</div>

<br/>

> Um dashboard acadêmico interativo de alto desempenho projetado para analisar e explorar **Mapas Auto-Organizáveis de Kohonen (SOM)** aplicados a duas frentes distintas: **séries temporais sintéticas** e **agrupamento semântico de textos**.
>
> Desenvolvido com uma estética premium baseada no tema **Tokyo Night**, o sistema possui controles de visualização completos, animações suaves e um pipeline híbrido de inteligência artificial (com execução local ou 100% serverless via nuvem).

<br/>

## `> stack_tecnológica`

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

**🐍 Backend / IA**<br/><br/>
<img src="https://img.shields.io/badge/FastAPI-Inferência_Local-c0caf5?style=flat-square&logo=fastapi&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Sentence--BERT-Embeddings_Semânticos-f7768e?style=flat-square"/>
<img src="https://img.shields.io/badge/TF--IDF-Representação_Frequencial-7aa2f7?style=flat-square"/>

</td>
<td align="center" width="33%">

**🧩 Dados**<br/><br/>
<img src="https://img.shields.io/badge/Synthetic_Control-600_Séries-9ece6a?style=flat-square"/>
<img src="https://img.shields.io/badge/20_Newsgroups-400_Notícias-bb9af7?style=flat-square"/>
<img src="https://img.shields.io/badge/HuggingFace-Inference_API-c0caf5?style=flat-square&logo=huggingface&logoColor=1a1b26"/>

</td>
</tr>
</table>

<br/>

## `> visão_geral_do_sistema`

O dashboard está estruturado em duas áreas de análise principais:

<table align="center">
<tr>
<td width="50%">

### 📈 1. Séries Temporais (Synthetic Control)
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**Dataset:** 600 séries temporais do dataset *Synthetic Control* divididas em 6 classes de tendência (Normal, Cíclica, Tendência Crescente, Tendência Decrescente, Desvio para Cima e Desvio para Baixo).

**Mapas suportados:** Grades hexagonais de tamanhos **5x5, 7x7, 10x10, 12x12, 15x15 e 20x20**.

**Visualizações:**
- **U-Matrix** — distâncias entre neurônios vizinhos na malha
- **Classe Dominante** — classe de tendência majoritária por neurônio
- **Pureza** — coesão interna do neurônio
- **Métricas de Baseline** — comparação em tempo real (ARI, NMI, Silhueta, Davies-Bouldin, Calinski-Harabasz, Erro de Quantização, Erro Topográfico) com K-Means, PCA + K-Means, Clustering Aglomerativo e DBSCAN
- **Radar Chart** — trade-offs das dimensões dos mapas SOM

</td>
<td width="50%">

### 📰 2. Classificador Semântico de Textos (20 Newsgroups)
<p><img src="https://img.shields.io/badge/Status-✅_Completo-9ece6a?style=for-the-badge"/></p>

**Dataset:** 400 notícias do corpus *20 Newsgroups* distribuídas igualmente em 4 classes de interesse (Graphics, Baseball, Mideast e Space).

**Mapas hexagonais (10x10):**
- Comparação lado a lado entre representação clássica **TF-IDF (Frequencial)** e representação profunda **Sentence-BERT (Semântico)**
- Análise estatística evidenciando o ganho de ARI/NMI usando embeddings contextuais

**Classificador em tempo real:**
- Digite qualquer texto curto ou longo e projete-o na malha de Kohonen para encontrar o neurônio ativado (BMU) correspondente
- **Pipeline híbrido:** tenta o servidor FastAPI local (`http://127.0.0.1:8000`); se inativo, consome a **API de inferência pública do Hugging Face** e executa a projeção PCA de 384D para 20D em JavaScript diretamente no navegador

</td>
</tr>
</table>

<br/>

## `> estrutura_de_arquivos`

```
intrasom-kohonen/
│
├── 🌐 frontend/                     # Aplicação web React + TS
│   ├── 📁 public/data/              # Dados estáticos pré-computados (CDN)
│   │   ├── series.json              # 600 séries temporais
│   │   ├── som_models.json          # Malhas 5x5 a 20x20 para séries
│   │   ├── text_models.json         # Grades textuais (SBERT + TF-IDF)
│   │   └── pca_params.json          # Média e componentes do PCA (384D -> 20D)
│   └── 📁 src/
│       ├── 📁 components/           # HexGrid, TextHexGrid, ClassifierPanel, etc.
│       └── 📁 store/
│           └── useDashboardStore.ts # Estado global do dashboard e lógica de inferência
│
└── 🐍 src/                          # Scripts Python e Inteligência Artificial
    ├── api.py                       # Servidor local FastAPI para vetorização/projeção
    ├── text_som_clustering.py       # Pipeline de treinamento do SOM textual
    └── export_data_for_frontend.py  # Exportador dos resultados Python para JSONs estáticos
```

<br/>

## `> execução_e_desenvolvimento`

**Pré-requisitos:**
<img src="https://img.shields.io/badge/Node.js-v20+-9ece6a?style=flat-square&logo=node.js&logoColor=1a1b26"/>
<img src="https://img.shields.io/badge/Python-v3.10+-3776ab?style=flat-square&logo=python&logoColor=ffffff"/>

### 🌐 Frontend (React + Vite)

```bash
# 1. Navegue até a pasta do frontend
cd frontend

# 2. Instale as dependências
npm install

# 3. Inicie o servidor de desenvolvimento
npm run dev
# → http://localhost:5173
```

### 🐍 API Backend Local (Opcional)

Para habilitar inferência em tempo real com TF-IDF nativo ou processar embeddings localmente de forma veloz:

```bash
# Instale as dependências
pip install -r requirements.txt

# Inicie a API FastAPI a partir da raiz do projeto
python src/api.py
# → http://127.0.0.1:8000
```

> Assim que a API estiver ativa, o frontend muda o indicador de status para **Servidor Local: ATIVO** de forma automatizada.

<br/>

## `> como_re_treinar_os_modelos`

Para reproduzir os treinos com os novos parâmetros científicos do professor (PCA, 500 épocas, raio inicial 80%→1) ou reconstruir os arquivos JSON estáticos do frontend:

```bash
# 1. Garanta que as dependências do Python estejam instaladas
pip install -r requirements.txt

# 2. Treine todos os modelos base do Synthetic Control (6 tamanhos de mapa: HEX_toroid)
python src/train_som.py

# 3. Treine as variantes planares para TODOS os 6 tamanhos (HEX_planar / Toroid OFF)
python src/train_som_variants.py

# 4. Treine as variantes retangulares para TODOS os 6 tamanhos via MiniSom (RECT_planar)
python src/train_som_rect.py

# 5. Execute o estudo de sensibilidade multivariado de parâmetros
python src/train_parameter_study.py

# 6. Treine os 4 modelos semânticos textuais (20news e 6class, TF-IDF e SBERT)
python src/text_som_clustering.py

# 7. Exporte todos os arquivos estruturados (.parquet + JSONs) para o React
python src/export_data_for_frontend.py
```

<br/>

## `> arquitetura_de_motores`

* **Grade Hexagonal (`IntraSOM`):**
  Todos os mapas hexagonais (tanto em topologia toroidal quanto em topologia plana) para os 6 tamanhos de mapa (5x5 a 20x20) são treinados com o motor principal `intrasom`.
* **Grade Retangular (`MiniSom`):**
  A biblioteca `intrasom` possui duas limitações estruturais que impedem o uso nativo para malhas retangulares:
  1. O método `build_umatrix()` lança explicitamente `Exception("build_umatrix error: non hexagonal lattice not implemented!")`.
  2. O cálculo de distâncias retangulares em `Codebook._rect_dist_plan` possui um bug de implementação (passa um gerador para `np.array()`, gerando um objeto 0-d em vez de um array de distâncias).
  Por essas razões (e não por escolha estilística arbitrária), utilizamos o motor complementar **MiniSom** (`minisom==2.3.6`) para treinar a variante **RECT_planar** real em todas as dimensões com o algoritmo batch síncrono `train_batch_offline`, garantindo 100% de dados reais e eliminando qualquer fallback decorativo.

<br/>


## `> acessibilidade_e_detalhes_especiais`

<table align="center">
<tr>
<td width="33%">

**⌨️ Suporte a Teclado**
Toda a malha hexagonal é navegável usando a tecla `Tab`. O neurônio focado pode ser ativado usando `Enter` ou `Espaço`.

</td>
<td width="33%">

**🔊 Leitores de Tela**
Tags ARIA completas (`role="button"`, `aria-label`) descrevem o ID do neurônio, a quantidade de amostras associadas, a classe dominante e a pureza de cada hexágono.

</td>
<td width="33%">

**🔐 Segurança**
O token do Hugging Face foi removido do código cliente, sendo carregado via variáveis de ambiente (`VITE_HF_TOKEN`) com fallback seguro e tokenless de taxa limitada para uso anônimo.

</td>
</tr>
</table>

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
  <img src="https://img.shields.io/badge/Feito_com-React_⚛️_+_Python_🐍-c0caf5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Powered_by-IntraSOM_+_Sentence--BERT-7aa2f7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tema-Tokyo_Night_🌃-bb9af7?style=for-the-badge"/>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1b26&height=100&section=footer"/>
