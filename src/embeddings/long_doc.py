from typing import List, Optional
import numpy as np

def chunk_words(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    words = str(text or '').split()
    if not words: return ['']
    step = max(1, chunk_size - overlap)
    chunks = [' '.join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
    return chunks if chunks else ['']

def embed_long_document(text: str, model, chunk_size: int = 220, overlap: int = 40, pooling: str = 'mean', weights: Optional[List[float]] = None) -> np.ndarray:
    chunks = chunk_words(text, chunk_size=chunk_size, overlap=overlap)
    chunk_embs = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    if len(chunk_embs.shape) == 1:
        chunk_embs = np.expand_dims(chunk_embs, axis=0)
    if pooling == 'max': doc_emb = np.max(chunk_embs, axis=0)
    elif pooling == 'weighted' and weights is not None and len(weights) == len(chunk_embs):
        w = np.array(weights)[:, np.newaxis]
        doc_emb = np.sum(chunk_embs * w, axis=0) / np.sum(weights)
    else: doc_emb = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(doc_emb)
    if norm > 0: doc_emb = doc_emb / norm
    return doc_emb.astype(np.float32)

def embed_zoned_document(zones: dict, model, zone_weights: Optional[dict] = None) -> np.ndarray:
    if zone_weights is None: zone_weights = {'title': 2.0, 'header': 1.0, 'recitals': 1.0, 'main_body': 1.5}
    zone_embs, w_list = [], []
    for zone_name, zone_text in zones.items():
        if not zone_text or not str(zone_text).strip(): continue
        emb = embed_long_document(zone_text, model, chunk_size=200, overlap=30, pooling='mean')
        zone_embs.append(emb)
        w_list.append(zone_weights.get(zone_name, 1.0))
    if not zone_embs:
        dim = getattr(model, 'get_sentence_embedding_dimension', lambda: 384)()
        return np.zeros((dim,), dtype=np.float32)
    embs_arr = np.array(zone_embs)
    weights_arr = np.array(w_list)[:, np.newaxis]
    doc_emb = np.sum(embs_arr * weights_arr, axis=0) / np.sum(weights_arr)
    norm = np.linalg.norm(doc_emb)
    if norm > 0: doc_emb = doc_emb / norm
    return doc_emb.astype(np.float32)
