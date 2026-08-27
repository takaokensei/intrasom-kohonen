"""
Script de download e cache local para HuffPost News Category e EUR-Lex 57K.
Baixa os parquets otimizados para data/text/ permitindo treinamento 100% offline.
"""

import os
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_DATA_DIR = os.path.join(BASE_DIR, "data", "text")
os.makedirs(TEXT_DATA_DIR, exist_ok=True)

HUFFPOST_URL = "https://huggingface.co/api/datasets/heegyu/news-category-dataset/parquet/default/train/0.parquet"
EURLEX_TRAIN_URL = "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/eurlex/train/0.parquet"
EURLEX_TEST_URL = "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/eurlex/test/0.parquet"

def download_and_cache():
    print("=" * 70)
    print("DOWNLOAD E CACHE DAS BASES HUFFPOST E EUR-LEX")
    print("=" * 70)

    # 1. HuffPost
    huff_target = os.path.join(TEXT_DATA_DIR, "huffpost_dataset.parquet")
    if os.path.exists(huff_target):
        print(f"HuffPost ja existe em: {huff_target}")
    else:
        print(f"Baixando HuffPost (~209k noticias) de {HUFFPOST_URL}...")
        df_huff = pd.read_parquet(HUFFPOST_URL)
        df_huff.to_parquet(huff_target)
        print(f"  Salvo com sucesso: {huff_target} (shape={df_huff.shape})")

    # 2. EUR-Lex (train + test)
    eurlex_target = os.path.join(TEXT_DATA_DIR, "eurlex_dataset.parquet")
    if os.path.exists(eurlex_target):
        print(f"EUR-Lex ja existe em: {eurlex_target}")
    else:
        print(f"Baixando EUR-Lex de {EURLEX_TRAIN_URL}...")
        df_eur_train = pd.read_parquet(EURLEX_TRAIN_URL)
        df_eur_test = pd.read_parquet(EURLEX_TEST_URL)
        df_eur_combined = pd.concat([df_eur_train, df_eur_test], ignore_index=True)
        df_eur_combined.to_parquet(eurlex_target)
        print(f"  Salvo com sucesso: {eurlex_target} (shape={df_eur_combined.shape})")

    print("\nTodos os datasets estao baixados e prontos em data/text/!")

if __name__ == "__main__":
    download_and_cache()
