# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""Shared pytest fixtures loading the committed reference dataset."""
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def reference_dataset():
    """Load the committed 2000-point, 3-class reference dataset."""
    data = np.load(DATA_DIR / "sample_3class.npz")
    return (
        data["xyz"].astype(np.float64),
        data["true_labels"].astype(np.int32),
        data["pred_labels"].astype(np.int32),
    )
