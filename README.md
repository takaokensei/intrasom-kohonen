# 🧠 IntraSOM Kohonen Maps Analyzer

<div align="center">
  <samp>Visualização Interativa · Mapas Auto-Organizáveis (SOM) · Séries Temporais · Processamento de Linguagem Natural</samp>
  <br/><br/>

  <img src="https://img.shields.io/badge/React-19-7aa2f7?style=for-the-badge&logo=react&logoColor=1a1b26"/>
  <img src="https://img.shields.io/badge/TypeScript-Strict_Mode-3178c6?style=for-the-badge&logo=typescript&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Vite-Built_System-646cff?style=for-the-badge&logo=vite&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=ffffff"/>
  <img src="https://img.shields.io/badge/Status-Concluído-9ece6a?style=for-the-badge"/>
</div>

---

> Um dashboard acadêmico interativo de alto desempenho projetado para analisar e explorar **Mapas Auto-Organizáveis de Kohonen (SOM)** aplicados a duas frentes distintas: **séries temporais sintéticas** e **agrupamento semântico de textos**. 
>
> Desenvolvido com uma estética premium baseada no tema **Tokyo Night**, o sistema possui controles de visualização completos, animações suaves e um pipeline híbrido de inteligência artificial (com execução local ou 100% serverless via nuvem).

---

## 🗺️ Visão Geral do Sistema

O dashboard está estruturado em duas áreas de análise principais:

### 1. Séries Temporais (Synthetic Control)
* **Dataset:** 600 séries temporais do dataset *Synthetic Control* divididas em 6 classes de tendência (Normal, Cíclica, Tendência Crescente, Tendência Decrescente, Desvio para Cima e Desvio para Baixo).
* **Mapas Suportados:** Grades hexagonais de tamanhos **5x5, 7x7, 10x10, 12x12, 15x15 e 20x20**.
* **Visualizações:**
  * **U-Matrix:** Exibição gráfica das distâncias entre neurônios vizinhos na malha.
  * **Classe Dominante:** Identificação da classe de tendência majoritária associada a cada neurônio.
  * **Pureza:** Métrica de coesão interna do neurônio.
  * **Métricas de Baseline:** Comparação em tempo real (ARI, NMI, Silhueta, Davies-Bouldin, Calinski-Harabasz, Erro de Quantização, Erro Topográfico) com algoritmos clássicos como K-Means, PCA + K-Means, Clustering Aglomerativo e DBSCAN.
  * **Radar Chart:** Exibição de trade-offs das dimensões dos mapas SOM.

### 2. Classificador Semântico de Textos (20 Newsgroups)
* **Dataset:** 400 notícias do famoso corpus *20 Newsgroups* distribuídas igualmente em 4 classes de interesse (Graphics, Baseball, Mideast e Space).
* **Mapas Hexagonais (10x10):**
  * Comparação lado a lado entre representação clássica **TF-IDF (Frequencial)** e representação profunda **Sentence-BERT (Semântico)**.
  * Análise estatística evidenciando o ganho de ARI/NMI usando embeddings contextuais.
* **Classificador em Tempo Real:**
  * Permite digitar qualquer texto curto ou longo e projetá-lo na malha de Kohonen para encontrar o neurônio ativado (BMU) correspondente.
  * Pipeline híbrido: tenta o servidor FastAPI local (`http://127.0.0.1:8000`); se inativo, consome a **API de inferência pública do Hugging Face** e executa a projeção PCA de 384D para 20D em JavaScript diretamente no navegador.

---

## 📁 Estrutura de Arquivos Principal

```text
intrasom-kohonen/
│
├── 📁 frontend/                     # Aplicação web React + TS
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
└── 📁 src/                          # Scripts Python e Inteligência Artificial
    ├── api.py                       # Servidor local FastAPI para vetorização/projeção
    ├── text_som_clustering.py       # Pipeline de treinamento do SOM textual
    └── export_data_for_frontend.py  # Exportador dos resultados Python para JSONs estáticos
```

---

## ⚡ Execução e Desenvolvimento

### Pré-requisitos
* Node.js v20+
* Python v3.10+ com dependências instaladas (`pip install -r requirements.txt`)

### 🌐 Frontend (React + Vite)
1. Navegue até a pasta do frontend:
   ```bash
   cd frontend
   ```
2. Instale as dependências:
   ```bash
   npm install
   ```
3. Inicie o servidor de desenvolvimento:
   ```bash
   npm run dev
   ```
4. O projeto estará rodando em: `http://localhost:5173`.

### 🐍 API Backend Local (Opcional)
Para habilitar inferência em tempo real com TF-IDF nativo ou processar embeddings localmente de forma veloz:
1. Inicie a API FastAPI a partir da raiz do projeto:
   ```bash
   python src/api.py
   ```
2. A API rodará em `http://127.0.0.1:8000`. O frontend mudará o indicador de status para **Servidor Local: ATIVO** de forma automatizada.

---

## 🎯 Acessibilidade & Detalhes Especiais
* **Suporte à Teclado:** Toda a malha hexagonal é navegável usando a tecla `Tab`. O neurônio focado pode ser ativado usando `Enter` ou `Espaço`.
* **Leitores de Tela:** Tags ARIA completas (`role="button"`, `aria-label`) descrevem o ID do neurônio, a quantidade de amostras associadas, a classe dominante e a pureza de cada hexágono.
* **Segurança:** O token do Hugging Face foi removido do código cliente, sendo carregado via variáveis de ambiente (`VITE_HF_TOKEN`) com fallback seguro e tokenless de taxa limitada para uso anônimo.
