import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional

TEXT_FIELDS = ["title", "header", "recitals", "main_body"]

def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)

def load_eurlex57k(root_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega o dataset EURLEX57K a partir da pasta de splits.
    """
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parquet_path = os.path.join(workspace_dir, "data", "text", "eurlex_dataset.parquet")
    if os.path.exists(parquet_path):
        print(f"Carregando EUR-Lex do parquet otimizado: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        print(f"EURLEX57K: Carregados {len(df)} documentos.")
        return df

    if root_dir is None:
        root_dir = os.path.join(workspace_dir, "data", "text", "EURLEX57K")

    root = Path(root_dir)
    if not root.exists():
        print(f"Aviso: EURLEX57K nao encontrado em {root_dir} nem em {parquet_path}.")
        return pd.DataFrame()

    rows = []
    for split in ["train", "dev", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for fp in split_dir.rglob("*.json"):
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            zones = {field: to_text(obj.get(field, "")).strip() for field in TEXT_FIELDS}
            full_text = "\n".join(zones[field] for field in TEXT_FIELDS if zones[field]).strip()
            labels = obj.get("concepts", []) or obj.get("labels", [])

            rows.append({
                "split": split,
                "file": str(fp),
                "celex_id": obj.get("celex_id", fp.stem),
                "text": full_text,
                "zones": zones,
                "labels": labels,
            })

    df = pd.DataFrame(rows)
    print(f"EURLEX57K: Carregados {len(df)} documentos.")
    return df
