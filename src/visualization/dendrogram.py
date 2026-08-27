"""
Dendrograma e Agrupamento Hierarquico dos Prototipos do SOM (Codigo 23 do Guia).
"""
import numpy as np
from scipy.cluster.hierarchy import linkage

def compute_som_prototype_linkage(
    codebook_matrix: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean"
) -> np.ndarray:
    return linkage(codebook_matrix, method=method, metric=metric)
