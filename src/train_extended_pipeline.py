"""
Pipeline unificado para execucao de experimentos com as 3 bases estendidas:
20 Newsgroups (20 classes), HuffPost News Category e EURLEX57K.
Baseado nas secoes 14, 15, 17 e 24 do Guia de Bases Textuais para IntraSOM.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.text_data import load_20news_full, load_6class_data
from src.data_loaders.huffpost import load_huffpost_data
from src.data_loaders.eurlex import load_eurlex57k
from src.metrics.entropy import calculate_local_neuron_entropy
from src.graph.som_graph import build_som_neuron_graph, compute_som_communities
from src.visualization.dendrogram import compute_som_prototype_linkage
from src.visualization.huffpost_temporal import compute_temporal_category_trajectories, compute_semantic_drift_distances
from src.search.semantic_search import search_semantic_global, search_topological_som

def run_dataset_audit():
    """Realiza auditoria dos datasets disponiveis e relata contagem de documentos."""
    print("=" * 70)
    print("AUDITORIA DAS BASES DE DADOS TEXTUAIS (20NG, HuffPost, EURLEX57K, 6Class)")
    print("=" * 70)

    # 1. 20NG Full
    try:
        docs_20ng, labels_20ng = load_20news_full(subset="all")
        print(f"20 Newsgroups Full: {len(docs_20ng)} documentos em {len(np.unique(labels_20ng))} categorias.")
    except Exception as e:
        print(f"20 Newsgroups: Erro ao carregar ({e})")

    # 2. HuffPost
    df_huff = load_huffpost_data(max_samples=1000)
    if not df_huff.empty:
        print(f"HuffPost: Disponivel com {len(df_huff)} amostras auditadas.")
    else:
        print("HuffPost: Arquivo bruto 'News_Category_Dataset_v3.json' pronto para ser colocado em 'data/text/'.")

    # 3. EURLEX57K
    df_eurlex = load_eurlex57k()
    if not df_eurlex.empty:
        print(f"EURLEX57K: Disponivel com {len(df_eurlex)} documentos legislativos.")
    else:
        print("EURLEX57K: Estrutura pronta para extracao do zip em 'data/text/EURLEX57K'.")

    # 4. 6Class PT
    docs_6c, labels_6c = load_6class_data()
    print(f"6Class PT: {len(docs_6c)} documentos em {len(np.unique(labels_6c))} categorias.")
    print("=" * 70)

if __name__ == "__main__":
    run_dataset_audit()
