"""
Grafo de Coocorrencia EUROVOC para EURLEX57K (Codigo 22 do Guia).
"""
from itertools import combinations
from collections import Counter
import networkx as nx
from typing import List

def build_eurovoc_coocurrence_graph(
    labels_multilabel: List[List[str]],
    min_coocurrence: int = 5
) -> nx.Graph:
    co = Counter()
    freq = Counter()

    for labels in labels_multilabel:
        unique_labels = sorted(set(labels))
        freq.update(unique_labels)
        for a, b in combinations(unique_labels, 2):
            co[(a, b)] += 1

    G_labels = nx.Graph()
    for lab, f in freq.items():
        G_labels.add_node(lab, frequency=f)

    for (a, b), weight in co.items():
        if weight >= min_coocurrence:
            G_labels.add_edge(a, b, weight=weight)

    return G_labels
