"""No-split: all data is used for both training and evaluation (unsupervised models)."""

from __future__ import annotations

import numpy as np
from anomaly_detect.data.splitters.base_splitter import BaseSplitter


class NoneSplitter(BaseSplitter):
    """Use the full dataset for training; test indices = all indices.
    Val is empty — no early stopping available with this splitter.
    Intended for unsupervised models (IsolationForest, Autoencoder).
    """

    def split(
        self,
        n_samples: int,
        y: np.ndarray = None,
        X: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_idx = np.arange(n_samples)
        return all_idx, np.array([], dtype=int), all_idx
