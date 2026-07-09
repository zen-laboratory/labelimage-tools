from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def sample_path() -> Path:
    path = ROOT / "samples" / "test_cells2D.tif"
    if not path.is_file():
        pytest.skip("sample label image missing")
    return path


@pytest.fixture
def simple_labels() -> np.ndarray:
    return np.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def nonconsecutive_labels() -> np.ndarray:
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[1:4, 1:4] = 5
    labels[4:7, 4:7] = 10
    return labels
