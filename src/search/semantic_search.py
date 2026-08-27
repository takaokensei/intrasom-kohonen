"""
Busca Semantica Global vs. Busca Topologica via BMU (Secao 12 do Guia).
"""
import numpy as np
from typing import List, Tuple

def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    if len(query_vec.shape) == 1:
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    return np.dot(doc_matrix, query_vec)

def search_semantic_global(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    sims = cosine_similarity_matrix(query_emb, doc_embs)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(int(idx), float(sims[idx])) for idx in top_indices]

def search_topological_som(
    query_emb: np.ndarray,
    codebook_matrix: np.ndarray,
    mapsize: Tuple[int, int],
    results_df,
    doc_embs: np.ndarray,
    top_k: int = 10,
    radius: int = 1
) -> List[Tuple[int, float]]:
    dists = np.linalg.norm(codebook_matrix - query_emb, axis=1)
    bmu_idx = int(np.argmin(dists))
    cols, rows = mapsize[0], mapsize[1]
    r_q = bmu_idx // cols
    c_q = bmu_idx % cols

    candidate_bmus = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr = r_q + dr
            nc = c_q + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                candidate_bmus.add(nr * cols + nc + 1)

    candidate_doc_indices = results_df[results_df["BMU"].isin(candidate_bmus)].index.values
    if len(candidate_doc_indices) == 0:
        return search_semantic_global(query_emb, doc_embs, top_k=top_k)

    candidate_embs = doc_embs[candidate_doc_indices]
    sims = cosine_similarity_matrix(query_emb, candidate_embs)
    rel_top = np.argsort(sims)[::-1][:top_k]
    
    return [(int(candidate_doc_indices[i]), float(sims[i])) for i in rel_top]
