"""
Grafo de Neuronios do SOM e Deteccao de Comunidades (Codigo 20 e 21 do Guia).
"""
import networkx as nx
import numpy as np
from typing import Tuple, Dict

def build_som_neuron_graph(
    codebook_matrix: np.ndarray,
    mapsize: Tuple[int, int],
    hits: np.ndarray = None,
    lattice: str = "hex"
) -> nx.Graph:
    cols, rows = mapsize[0], mapsize[1]
    W = codebook_matrix.reshape(rows, cols, -1)
    G = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            bmu_id = r * cols + c + 1
            node_hits = int(hits[r, c]) if hits is not None else 0
            G.add_node((r, c), bmu=bmu_id, hits=node_hits)

            neighbors = [(r + 1, c), (r, c + 1)]
            if lattice == "hex" and r % 2 == 1:
                neighbors.extend([(r + 1, c + 1), (r + 1, c - 1)])

            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:
                    d = float(np.linalg.norm(W[r, c] - W[nr, nc]))
                    sim = float(1.0 / (1.0 + d))
                    G.add_edge((r, c), (nr, nc), distance=d, similarity=sim, weight=sim)

    return G

def compute_som_communities(G: nx.Graph) -> Dict[Tuple[int, int], int]:
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight="similarity"))
    node_to_comm = {}
    for cid, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = cid
    return node_to_comm
