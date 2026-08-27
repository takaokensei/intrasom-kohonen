"""
Analise Temporal e Semantic Drift no HuffPost (Codigo 18 do Guia).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def compute_temporal_category_trajectories(
    df_huff: pd.DataFrame,
    X_emb: np.ndarray,
    target_categories: List[str] = None,
    min_samples_per_year: int = 15
) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    if target_categories is None:
        target_categories = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "TECH", "WORLD NEWS"]

    sub = df_huff.loc[df_huff["year"].notna()].copy()
    trajectories = {}

    for category in target_categories:
        pts = []
        for year in sorted(sub["year"].dropna().unique()):
            cat_mask = (sub["category"] == category) & (sub["year"] == year)
            ids = np.where(cat_mask.values)[0]
            if len(ids) >= min_samples_per_year:
                centroid = X_emb[ids].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                pts.append((int(year), centroid))
        trajectories[category] = pts

    return trajectories

def compute_semantic_drift_distances(trajectories: Dict[str, List[Tuple[int, np.ndarray]]]) -> Dict[str, float]:
    drift = {}
    for cat, pts in trajectories.items():
        if len(pts) >= 2:
            first_pt = pts[0][1]
            last_pt = pts[-1][1]
            drift[cat] = float(np.linalg.norm(last_pt - first_pt))
    return drift
