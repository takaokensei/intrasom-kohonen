"""
Módulo compartilhado de carregamento de datasets de texto.
Consumido por text_som_clustering.py, train_text_som_rect.py e export_data_for_frontend.py.
Extraído de text_som_clustering.py para eliminar duplicação.
"""

import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups


def load_20news_data():
    """Loads a subset of 20 Newsgroups with 4 distinct categories (400 docs)."""
    print("Fetching 20 Newsgroups subset...")
    categories = [
        'comp.graphics',
        'sci.space',
        'rec.sport.baseball',
        'talk.politics.mideast'
    ]
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    docs = []
    labels = []
    class_map = {
        'comp.graphics':         'Graphics',
        'sci.space':             'Space',
        'rec.sport.baseball':    'Baseball',
        'talk.politics.mideast': 'Mideast'
    }
    counts = {cat: 0 for cat in categories}
    for text, label_idx in zip(newsgroups.data, newsgroups.target):
        cat_name = newsgroups.target_names[label_idx]
        if counts[cat_name] < 100 and len(text.strip()) > 50:
            docs.append(text)
            labels.append(class_map[cat_name])
            counts[cat_name] += 1
    return docs, np.array(labels)


def load_6class_data():
    """Loads the 6-class Brazilian Portuguese news dataset (317 docs, 6 categories)."""
    import pandas as pd
    print("Loading 6-class Excel text dataset...")
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(workspace_dir, "data", "text", "base_dados_textos_6_classes.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset não encontrado em: {file_path}\n"
            "Copie o arquivo 'Base_dados_textos_6_classes.xlsx' para a pasta "
            "'data/text/' na raiz do projeto.\n"
            "Renomeie para: base_dados_textos_6_classes.xlsx (sem espaços/acentos)."
        )
    df = pd.read_excel(file_path)

    clean_categories = []
    for cat in df['Categoria']:
        c = str(cat).replace("Polcia", "Policia").replace("Polícia", "Policia").replace("Polé​cia", "Policia")
        c = c.replace("Polícia", "Policia")
        c = c.replace("Política", "Politica").replace("Poltica", "Politica")
        c = c.replace("Polícia e Direitos", "Policia").replace("Política", "Politica")
        c = c.split(" e ")[0]

        if "Turismo" in c:
            clean_categories.append("Turismo")
        elif "Esporte" in c:
            clean_categories.append("Esportes")
        elif "Polici" in c or "Policia" in c:
            clean_categories.append("Policia")
        elif "Economia" in c:
            clean_categories.append("Economia")
        elif "Politic" in c or "Politica" in c:
            clean_categories.append("Politica")
        elif "Variedades" in c or "Sociedade" in c:
            clean_categories.append("Variedades")
        else:
            clean_categories.append(c)

    df['CleanCategory'] = clean_categories
    docs = df['Texto Expandido'].fillna(df['Texto Original']).fillna('').astype(str).tolist()
    labels = np.array(df['CleanCategory'].tolist())
    return docs, labels
