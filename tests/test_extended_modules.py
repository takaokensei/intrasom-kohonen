import unittest
import numpy as np
import pandas as pd
from src.embeddings.long_doc import chunk_words, embed_long_document, embed_zoned_document
from src.metrics.entropy import calculate_local_neuron_entropy
from src.graph.som_graph import build_som_neuron_graph, compute_som_communities
from src.graph.eurovoc_coocurrence import build_eurovoc_coocurrence_graph
from src.search.semantic_search import search_semantic_global, search_topological_som
from src.visualization.huffpost_temporal import compute_temporal_category_trajectories, compute_semantic_drift_distances
from src.visualization.dendrogram import compute_som_prototype_linkage

class MockSentenceTransformer:
    def __init__(self, dim=384):
        self.dim = dim
    def get_sentence_embedding_dimension(self):
        return self.dim
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

class TestExtendedModules(unittest.TestCase):
    def test_long_doc_chunking(self):
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = chunk_words(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= 100 for c in chunks))

    def test_long_doc_embedding(self):
        model = MockSentenceTransformer(dim=64)
        text = " ".join([f"word{i}" for i in range(300)])
        emb = embed_long_document(text, model, chunk_size=100, overlap=20)
        self.assertEqual(emb.shape, (64,))
        self.assertTrue(np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5))

    def test_zoned_document_embedding(self):
        model = MockSentenceTransformer(dim=64)
        zones = {"title": "Law on digital markets", "main_body": "Full legal text here."}
        emb = embed_zoned_document(zones, model)
        self.assertEqual(emb.shape, (64,))
        self.assertTrue(np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5))

    def test_local_neuron_entropy(self):
        results_df = pd.DataFrame({"BMU": [1, 1, 2, 3]})
        labels = np.array(["catA", "catA", "catB", "catC"])
        entropy_grid, entropy_dict = calculate_local_neuron_entropy(results_df, labels, (2, 2))
        self.assertEqual(entropy_grid.shape, (2, 2))
        self.assertEqual(entropy_dict[1], 0.0)
        self.assertEqual(entropy_dict[4], 0.0)

    def test_som_neuron_graph_and_communities(self):
        codebook = np.random.randn(9, 16)
        G = build_som_neuron_graph(codebook, (3, 3), lattice="hex")
        self.assertEqual(G.number_of_nodes(), 9)
        self.assertGreater(G.number_of_edges(), 0)
        comm = compute_som_communities(G)
        self.assertEqual(len(comm), 9)

    def test_eurovoc_coocurrence_graph(self):
        multilabels = [["law", "tech", "eu"], ["law", "finance"], ["tech", "ai", "eu"]]
        G = build_eurovoc_coocurrence_graph(multilabels, min_coocurrence=1)
        self.assertTrue(G.has_edge("law", "tech"))
        self.assertTrue(G.has_edge("tech", "eu"))

    def test_semantic_and_topological_search(self):
        rng = np.random.default_rng(42)
        docs = rng.standard_normal((20, 32)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        query = rng.standard_normal((32,)).astype(np.float32)
        query /= np.linalg.norm(query)
        
        res_glob = search_semantic_global(query, docs, top_k=5)
        self.assertEqual(len(res_glob), 5)
        self.assertGreaterEqual(res_glob[0][1], res_glob[1][1])

        codebook = rng.standard_normal((4, 32)).astype(np.float32)
        codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
        results_df = pd.DataFrame({"BMU": [1, 1, 2, 2, 3, 3, 4, 4] * 2 + [1, 2, 3, 4]})
        res_top = search_topological_som(query, codebook, (2, 2), results_df, docs, top_k=3, radius=1)
        self.assertLessEqual(len(res_top), 3)

    def test_huffpost_temporal_trajectories(self):
        df = pd.DataFrame({
            "category": ["POLITICS"] * 30 + ["TECH"] * 30,
            "year": [2015] * 15 + [2020] * 15 + [2015] * 15 + [2020] * 15,
            "text": ["some text"] * 60
        })
        X_emb = np.random.randn(60, 32)
        trajs = compute_temporal_category_trajectories(df, X_emb, ["POLITICS", "TECH"], min_samples_per_year=10)
        self.assertIn("POLITICS", trajs)
        self.assertEqual(len(trajs["POLITICS"]), 2)
        drift = compute_semantic_drift_distances(trajs)
        self.assertIn("POLITICS", drift)
        self.assertGreaterEqual(drift["POLITICS"], 0.0)

    def test_dendrogram_linkage(self):
        codebook = np.random.randn(10, 16)
        Z = compute_som_prototype_linkage(codebook)
        self.assertEqual(Z.shape, (9, 4))

if __name__ == "__main__":
    unittest.main()
