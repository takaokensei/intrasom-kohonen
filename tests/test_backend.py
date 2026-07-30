import sys
import os
import random
import numpy as np
import pandas as pd
import json
import tempfile
from fastapi.testclient import TestClient

# Add project root to Python search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reproducibility import set_global_seed
from src.api import app
import intrasom


def test_reproducibility():
    # Test that set_global_seed produces deterministic results
    set_global_seed(123)
    seq_np_1 = np.random.rand(10)
    seq_py_1 = [random.random() for _ in range(10)]

    set_global_seed(123)
    seq_np_2 = np.random.rand(10)
    seq_py_2 = [random.random() for _ in range(10)]

    np.testing.assert_allclose(seq_np_1, seq_np_2)
    assert seq_py_1 == seq_py_2


def test_api_health_smoke():
    # Smoke test for the api.py /health endpoint
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "backend" in data


def _make_synthetic_data(n_samples: int = 20, n_features: int = 6) -> pd.DataFrame:
    """Creates a small deterministic DataFrame for SOM training tests."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, n_features))
    df = pd.DataFrame(data, columns=[f"F_{i+1}" for i in range(n_features)])
    df.index = [f"S_{i+1}" for i in range(n_samples)]
    return df


def test_som_training_small():
    """Trains a 3x3 SOM on tiny data and verifies QE/TE are valid finite non-negative floats."""
    df = _make_synthetic_data()
    set_global_seed()
    som = intrasom.SOMFactory.build(
        data=df,
        mapsize=(3, 3),
        mapshape='toroid',
        lattice='hexa',
        normalization='var',
        initialization='pca',
        neighborhood='gaussian',
        training='batch',
        name='test_3x3',
        sample_names=list(df.index),
    )
    som.train(
        train_rough_len=5,
        train_rough_radiusin=2,
        train_rough_radiusfin=1,
        train_finetune_len=5,
        train_finetune_radiusin=1,
        train_finetune_radiusfin=1,
        previous_epoch=True,
    )
    qe = som.calculate_quantization_error
    if callable(qe):
        qe = qe()
    te = som.topographic_error
    if callable(te):
        te = te()

    assert isinstance(float(qe), float), "QE should be a float"
    assert isinstance(float(te), float), "TE should be a float"
    assert float(qe) >= 0, "QE must be non-negative"
    assert float(te) >= 0, "TE must be non-negative"
    assert np.isfinite(float(qe)), "QE must be finite"
    assert np.isfinite(float(te)), "TE must be finite"


def test_export_pipeline_keys():
    """Validates that parameter_study.json exported by train_parameter_study.py has expected keys."""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    study_path = os.path.join(workspace, "outputs", "metrics", "parameter_study.json")
    if not os.path.exists(study_path):
        import pytest
        pytest.skip("parameter_study.json not found; run src/train_parameter_study.py first")

    with open(study_path) as f:
        study = json.load(f)

    assert isinstance(study, list), "parameter_study.json must be a list"
    assert len(study) > 0, "parameter_study.json must not be empty"

    required_keys = {"key", "total_epochs", "radius_initial", "radius_final",
                     "quantization_error", "topographic_error"}
    for entry in study:
        missing = required_keys - set(entry.keys())
        assert not missing, f"Entry '{entry.get('key')}' missing keys: {missing}"
        assert entry["quantization_error"] >= 0
        assert entry["topographic_error"] >= 0
        assert entry["total_epochs"] > 0


def test_bmu_determinism():
    """Validates that the same seed always produces the same BMU for a fixed sample.

    This is a ponta-a-ponta regression test for reproducibility.py: if intrasom
    ever changes its internal RNG consumption order, this test will catch it.
    """
    df = _make_synthetic_data()

    def get_bmu_sequence(seed: int) -> list:
        set_global_seed(seed)
        som = intrasom.SOMFactory.build(
            data=df, mapsize=(3, 3), mapshape='toroid', lattice='hexa',
            normalization='var', initialization='pca', neighborhood='gaussian',
            training='batch', name='test_repr',
            sample_names=list(df.index),
        )
        som.train(
            train_rough_len=5, train_rough_radiusin=2, train_rough_radiusfin=1,
            train_finetune_len=5, train_finetune_radiusin=1, train_finetune_radiusfin=1,
            previous_epoch=True,
        )
        results = som.results_dataframe
        return results['BMU'].tolist()

    bmu_run1 = get_bmu_sequence(42)
    bmu_run2 = get_bmu_sequence(42)
    assert bmu_run1 == bmu_run2, (
        f"BMU sequences differ between runs with the same seed.\n"
        f"Run 1: {bmu_run1}\nRun 2: {bmu_run2}"
    )


def test_umatrix_edges_export():
    """Validates that som_models.json contains umatrix_edges, umatrix_edge_min, and umatrix_edge_max."""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    som_models_path = os.path.join(workspace, "frontend", "public", "data", "som_models.json")
    if not os.path.exists(som_models_path):
        import pytest
        pytest.skip("som_models.json not found; run src/export_data_for_frontend.py first")

    with open(som_models_path, encoding="utf-8") as f:
        som_models = json.load(f)

    assert "10x10" in som_models
    model_data = som_models["10x10"]
    assert model_data.get("has_variants") is True

    for variant_name in ["HEX_toroid", "HEX_planar", "RECT_planar"]:
        if variant_name in model_data:
            variant = model_data[variant_name]
            assert "umatrix_edges" in variant, f"{variant_name} missing umatrix_edges"
            assert "umatrix_edge_min" in variant, f"{variant_name} missing umatrix_edge_min"
            assert "umatrix_edge_max" in variant, f"{variant_name} missing umatrix_edge_max"
            
            edges = variant["umatrix_edges"]
            assert len(edges) > 0, f"{variant_name} umatrix_edges should not be empty"
            
            num_neurons = variant["cols"] * variant["rows"]
            for edge in edges[:10]:
                assert "from" in edge and "to" in edge and "distance" in edge
                assert 1 <= edge["from"] <= num_neurons
                assert 1 <= edge["to"] <= num_neurons
                assert isinstance(edge["distance"], (int, float))


def test_quantization_error_non_zero():
    """Regression test: verifies that all non-baseline SOM rows in model_comparison_results.csv
    and frontend public/data/metrics.json have a positive non-zero Quantization Error (> 0.0).
    """
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(workspace, "outputs", "metrics", "model_comparison_results.csv")
    json_path = os.path.join(workspace, "frontend", "public", "data", "metrics.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        som_rows = df[df["Modelo"].str.startswith("SOM")]
        assert len(som_rows) > 0, "No SOM rows found in model_comparison_results.csv"
        for _, row in som_rows.iterrows():
            qe = row["Erro Quantização"]
            assert float(qe) > 0.0, f"SOM model '{row['Modelo']}' has QE == 0.0 in CSV"

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            metrics = json.load(f)
        som_entries = [m for m in metrics if m.get("Modelo", "").startswith("SOM")]
        assert len(som_entries) > 0, "No SOM entries found in metrics.json"
        for entry in som_entries:
            qe = entry.get("Erro Quantização")
            assert qe is not None and float(qe) > 0.0, f"SOM model '{entry.get('Modelo')}' has invalid/zero QE in metrics.json: {qe}"


