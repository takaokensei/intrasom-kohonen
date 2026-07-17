import sys
import os
import random
import numpy as np
from fastapi.testclient import TestClient

# Add project root to Python search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reproducibility import set_global_seed
from src.api import app

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
