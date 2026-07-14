# Mapas de Kohonen com IntraSOM - Synthetic Control & Text Clustering

Este repositório contém a implementação completa dos experimentos para a disciplina/pesquisa do Prof. Alfredo (UFRN) sobre Mapas Auto-Organizáveis de Kohonen usando a biblioteca `IntraSOM/USP` aplicada ao dataset `Synthetic Control` da UCI e clusterização semântica de notícias.

## 📁 Estrutura do Projeto

```text
intrasom_synthetic_control/
|-- data/                          # Dataset original baixado da UCI
|-- notebooks/
|   |-- seminario_intrasom_synthetic_control.ipynb  # Notebook integrador completo
|-- src/
|   |-- load_data.py               # Leitura e normalização Z-score dos dados
|   |-- train_som.py               # Treinamento das redes de Kohonen (5x5 a 20x20)
|   |-- plot_som.py                # Visualizações (U-Matrix, Component Planes, etc.)
|   |-- evaluate_clusters.py       # Consolidação métrica quantitativa vs Baselines
|   |-- text_som_clustering.py     # Extensão textual (TF-IDF vs Sentence-BERT)
|-- outputs/
|   |-- figures/                   # Imagens geradas prontas para os slides
|   |-- maps/                      # Pesos (neurônios) e resultados dos modelos salvos
|   |-- metrics/                   # Planilhas de resultados quantitativos
|-- requirements.txt
```

## 🚀 Como Executar

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar todo o pipeline
Você pode executar cada módulo separadamente ou executar o notebook integrado em `notebooks/seminario_intrasom_synthetic_control.ipynb`.

Para rodar os scripts via terminal na sequência:
```bash
# 1. Baixar os dados e treinar os modelos SOM (5x5, 7x7, 10x10, 12x12, 15x15, 20x20)
python src/train_som.py

# 2. Gerar todas as figuras na pasta outputs/figures/
python src/plot_som.py

# 3. Gerar a planilha de métricas e comparação quantitativa contra K-Means
python src/evaluate_clusters.py

# 4. Rodar o comparativo de classificação textual (TF-IDF vs SBERT)
python src/text_som_clustering.py
```

## 📊 Principais Descobertas
*   **Melhor Desempenho:** Os mapas **7x7** e **10x10** atingiram a maior fidelidade aos grupos reais ($ARI = 0.66$, $NMI = 0.81$), superando significativamente o agrupamento tradicional do K-Means direto ($ARI = 0.56$).
*   **Extensão Textual:** O mapeamento espacial de notícias alimentado por embeddings **Sentence-BERT** obteve resultados semânticos bem superiores ($ARI = 0.27$, $NMI = 0.34$) se comparado ao saco de palavras **TF-IDF + LSA** ($ARI = 0.19$, $NMI = 0.26$).
*   **Reprodutibilidade:** Todos os resultados e métricas detalhados podem ser encontrados de forma sumarizada no relatório [walkthrough.md](../../.gemini/antigravity/brain/afcb98bc-a3ff-4b27-8fd5-4252e55e9121/walkthrough.md) ou visualizados interativamente no Notebook.
