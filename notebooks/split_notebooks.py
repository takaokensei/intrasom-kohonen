import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created {filename}")

def build_cells():
    # Common imports cell
    imports_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import json\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "# Garantir que a pasta src esteja no path\n",
            "sys.path.append(os.path.abspath(\"../src\"))\n",
            "from load_data import load_synthetic_control, CLASS_NAMES\n",
            "\n",
            "print(\"Ambiente importado com sucesso!\")"
        ]
    }
    
    # 1. EDA Notebook
    eda_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Análise Exploratória de Dados (EDA) - Synthetic Control Chart\n",
                "\n",
                "Este notebook executa a análise exploratória da base Synthetic Control da UCI, contendo 600 séries temporais divididas em 6 classes."
            ]
        },
        imports_cell,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X, y = load_synthetic_control(\"../data/synthetic_control.data\")\n",
                "print(f\"Dimensão dos atributos (X): {X.shape}\")\n",
                "print(f\"Dimensão dos rótulos reais (y): {y.shape}\")\n",
                "print(\"\\nDistribuição das amostras por classe:\")\n",
                "print(pd.Series(y).value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Visualização das Séries Temporais por Classe\n",
                "Exibimos o gráfico contendo o comportamento típico de cada uma das 6 classes de sinais da base."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 8), dpi=150)\n",
                "img = Image.open(\"../outputs/figures/time_series_classes.png\")\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Redução de Dimensionalidade Linear vs Não-Linear\n",
                "Comparação de projeções bidimensionais com PCA e t-SNE como referências externas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5), dpi=150)\n",
                "img = Image.open(\"../outputs/figures/dim_reduction_comparison.png\")\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        }
    ]
    
    # 2. Train Notebook
    train_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02. Treinamento dos Mapas Auto-Organizáveis (SOM)\n",
                "\n",
                "Neste notebook realizamos o treinamento dos mapas SOM com diferentes dimensões (de 5x5 a 20x20) e avaliamos os erros de representação topológica."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Treinamento dos Mapas\n",
                "Você pode treinar os mapas SOM executando o script `train_som.py` localizado na pasta `src/`:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Roda o script de treinamento se necessário\n",
                "# !python ../src/train_som.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Erros de Treinamento dos Mapas Kohonen\n",
                "Abaixo comparamos os erros de Quantização (QE) e erros Topográficos (TE) obtidos para cada tamanho de mapa treinado."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "errors_df = pd.read_csv(\"../outputs/metrics/som_error_comparison.csv\")\n",
                "errors_df"
            ]
        }
    ]
    
    # 3. Visualizations Notebook
    viz_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 03. Visualizações Estruturadas dos Mapas de Kohonen\n",
                "\n",
                "Neste notebook carregamos as visualizações clássicas do SOM geradas com a biblioteca IntraSOM: U-Matrices, Component Planes e Mapas de Hits."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### U-Matrix e Mapa de Hits (Grade 10x10)\n",
                "A U-Matrix ajuda a visualizar as fronteiras de distância na grade, enquanto o mapa de Hits mostra a densidade de amostras por neurônio."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150)\n",
                "axes[0].imshow(Image.open(\"../outputs/figures/umatrix_10x10.jpg\"))\n",
                "axes[0].axis('off')\n",
                "axes[0].set_title(\"U-Matrix 10x10\", fontsize=14, fontweight='bold')\n",
                "\n",
                "axes[1].imshow(Image.open(\"../outputs/figures/umatrix_hits_10x10_with_hits.jpg\"))\n",
                "axes[1].axis('off')\n",
                "axes[1].set_title(\"Mapa de Hits (Amostras)\", fontsize=14, fontweight='bold')\n",
                "\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Mapas de Classes Dominantes (Grades 15x15 e 20x20)\n",
                "Identificação da classe majoritária associada a cada neurônio receptor do mapa pós-treinamento."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=150)\n",
                "axes[0].imshow(Image.open(\"../outputs/figures/som_15x15_dominant_classes.png\"))\n",
                "axes[0].axis('off')\n",
                "axes[0].set_title(\"Classes Dominantes (15x15)\", fontsize=14, fontweight='bold')\n",
                "\n",
                "axes[1].imshow(Image.open(\"../outputs/figures/som_20x20_dominant_classes.png\"))\n",
                "axes[1].axis('off')\n",
                "axes[1].set_title(\"Classes Dominantes (20x20)\", fontsize=14, fontweight='bold')\n",
                "\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Component Planes (Fatias de Variáveis)\n",
                "Fatias do mapa de pesos sinápticos mostrando a ativação média do mapa em instantes de tempo específicos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 8), dpi=150)\n",
                "img = Image.open(\"../outputs/figures/Component_plots_collage_page1.jpg\")\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        }
    ]
    
    # 4. Cluster Evaluation Notebook
    eval_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 04. Agrupamento dos Neurônios e Avaliação Métricas\n",
                "\n",
                "Neste notebook avaliamos quantitativamente a qualidade dos agrupamentos dos neurônios do SOM em 6 classes correspondentes às classes reais."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tabela de Métricas de Agrupamento\n",
                "Comparação de ARI, NMI, Silhueta, Davies-Bouldin e Calinski-Harabasz de todos os tamanhos de mapas de Kohonen."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "comparison_df = pd.read_csv(\"../outputs/metrics/model_comparison_results.csv\")\n",
                "# Mostrar apenas as linhas referentes aos modelos SOM\n",
                "som_metrics = comparison_df[comparison_df['Modelo'].str.startswith('SOM')]\n",
                "som_metrics"
            ]
        }
    ]
    
    # 5. Baselines Comparison Notebook
    baseline_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 05. Comparação do SOM contra Algoritmos Clássicos (Baselines)\n",
                "\n",
                "Neste notebook comparamos o desempenho do SOM contra K-Means, PCA + K-Means, Agglomerative Clustering e DBSCAN."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tabela Completa de Métricas de Benchmarking\n",
                "Carregamos e analisamos a comparação final contendo todos os baselines."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "comparison_df = pd.read_csv(\"../outputs/metrics/model_comparison_results.csv\")\n",
                "comparison_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Análise de Desempenho:\n",
                "1. **SOM 5x5** e **SOM 10x10** apresentam os maiores valores de ARI e NMI (~0.65 e ~0.80).\n",
                "2. O **K-Means direto** e o **Agglomerative Clustering** alcançam boa coesão local (Silhouette de 0.27 e 0.25), mas erram mais o particionamento global em relação às classes reais (ARI de 0.56 e 0.61).\n",
                "3. O **DBSCAN** obteve baixa performance (ARI de 0.14) devido à alta dimensionalidade das séries que dispersa o conceito de densidade."
            ]
        }
    ]
    
    # 6. Text TF-IDF Notebook
    tfidf_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 06. Clusterização de Textos com SOM — TF-IDF + LSA\n",
                "\n",
                "Neste notebook avaliamos a representação clássica de textos com frequência de termos (TF-IDF) reduzida via LSA para treinamento do mapa auto-organizável."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Métricas de Agrupamento de Texto\n",
                "Visualizamos o desempenho do modelo TF-IDF contra o Sentence-BERT."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(\"../outputs/metrics/text_clustering_comparison.json\", \"r\") as f:\n",
                "    text_metrics = json.load(f)\n",
                "text_df = pd.DataFrame(text_metrics).T\n",
                "text_df.loc[['TF_IDF']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Organização Espacial com TF-IDF + LSA\n",
                "Exibimos a figura contendo a projeção dos textos no mapa de Kohonen 10x10 usando representação TF-IDF."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(9, 9), dpi=150)\n",
                "img = Image.open(\"../outputs/figures/som_text_tf-idf_dominant.png\")\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        }
    ]
    
    # 7. Text SBERT Notebook
    bert_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 07. Clusterização de Textos com SOM — Sentence-BERT Embeddings\n",
                "\n",
                "Neste notebook avaliamos a representação semântica contextual de notícias obtida através do Sentence-BERT (SBERT) mapeada no SOM."
            ]
        },
        imports_cell,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Métricas de Agrupamento de Texto\n",
                "Visualizamos a comparação de desempenho do modelo SBERT contra o TF-IDF."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(\"../outputs/metrics/text_clustering_comparison.json\", \"r\") as f:\n",
                "    text_metrics = json.load(f)\n",
                "text_df = pd.DataFrame(text_metrics).T\n",
                "text_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Organização Espacial com Sentence-BERT\n",
                "Exibimos a figura contendo a projeção dos textos no mapa de Kohonen 10x10 usando representação Sentence-BERT, mostrando alta coesão das classes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(9, 9), dpi=150)\n",
                "img = Image.open(\"../outputs/figures/som_text_sbert_dominant.png\")\n",
                "plt.imshow(img)\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        }
    ]
    
    notebooks_dir = "c:/IntraSOM_Kohonen_Synthetic_Control_Visual_Law/notebooks"
    os.makedirs(notebooks_dir, exist_ok=True)
    
    create_notebook(os.path.join(notebooks_dir, "01_eda_synthetic_control.ipynb"), eda_cells)
    create_notebook(os.path.join(notebooks_dir, "02_train_som_synthetic_control.ipynb"), train_cells)
    create_notebook(os.path.join(notebooks_dir, "03_visualizations_som.ipynb"), viz_cells)
    create_notebook(os.path.join(notebooks_dir, "04_cluster_evaluation.ipynb"), eval_cells)
    create_notebook(os.path.join(notebooks_dir, "05_baselines_comparison.ipynb"), baseline_cells)
    create_notebook(os.path.join(notebooks_dir, "06_text_news_som_tfidf.ipynb"), tfidf_cells)
    create_notebook(os.path.join(notebooks_dir, "07_text_news_som_bert.ipynb"), bert_cells)

if __name__ == "__main__":
    build_cells()
