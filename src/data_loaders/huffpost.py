import os
import json
import pandas as pd
import numpy as np
from typing import Optional

def load_huffpost_data(
    file_path: Optional[str] = None,
    max_samples: Optional[int] = 50000,
    min_category_samples: int = 50,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Carrega o dataset HuffPost News Category.
    """
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parquet_path = os.path.join(workspace_dir, "data", "text", "huffpost_dataset.parquet")
    if file_path is None:
        file_path = os.path.join(workspace_dir, "data", "text", "News_Category_Dataset_v3.json")

    if os.path.exists(parquet_path):
        print(f"Carregando HuffPost do parquet otimizado: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(file_path):
        print(f"Carregando HuffPost do JSON: {file_path}...")
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        print(f"Aviso: HuffPost nao encontrado em {parquet_path} nem em {file_path}.")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["text"] = (df["headline"].fillna("") + ". " + df["short_description"].fillna("")).str.strip()

    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= min_category_samples].index
    df = df[df["category"].isin(valid_cats)].copy()

    if max_samples is not None and len(df) > max_samples:
        df = df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(max_samples * len(x) / len(df))), random_state=random_state)
        ).reset_index(drop=True)

    print(f"  Carregadas {len(df)} noticias em {df['category'].nunique()} categorias.")
    return df
