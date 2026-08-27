"""
Calculo da Entropia Local do Mapa de Kohonen (Secao 7.2 do Guia).
Identifica nos/regioes onde diferentes classes se misturam (fronteiras semanticas).
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def calculate_local_neuron_entropy(
    results_df: pd.DataFrame,
    labels: np.ndarray,
    mapsize: Tuple[int, int]
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Calcula a entropia de Shannon da distribuicao de classes em cada neuronio do SOM:
    H(n) = - sum_c p_c(n) * log2(p_c(n))
    
    Returns:
        entropy_grid: Matriz 2D de shape (rows, cols)
        entropy_dict: Dicionario {bmu_id: entropy_value}
    """
    temp_df = results_df.copy()
    temp_df["Class"] = labels
    
    rows, cols = mapsize[1], mapsize[0]
    entropy_grid = np.zeros((rows, cols), dtype=float)
    entropy_dict = {}

    counts = temp_df.groupby(["BMU", "Class"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)

    for bmu in range(1, rows * cols + 1):
        if bmu in counts.index and totals[bmu] > 0:
            probs = counts.loc[bmu].values / totals[bmu]
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs))
        else:
            ent = 0.0

        r = (bmu - 1) // cols
        c = (bmu - 1) % cols
        entropy_grid[r, c] = ent
        entropy_dict[bmu] = float(ent)

    return entropy_grid, entropy_dict
