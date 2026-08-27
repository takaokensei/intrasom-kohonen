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

def analyze_eurovoc_modularity(G: nx.Graph) -> dict:
    if len(G) == 0:
        return {"num_nodes": 0, "num_edges": 0, "num_communities": 0, "modularity": 0.0}
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        comms = list(greedy_modularity_communities(G))
        q = float(modularity(G, comms)) if comms and len(G.edges) > 0 else 0.0
        return {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "num_communities": len(comms),
            "modularity": q
        }
    except Exception:
        return {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "num_communities": 1,
            "modularity": 0.0
        }

