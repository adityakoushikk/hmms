"""Stratified K-Fold splitter for supervised models (LogisticRegression)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from anomaly_detect.data.splitters.base_splitter import BaseSplitter


class StratifiedKFoldSplitter(BaseSplitter):
    """One fold used as test set; remaining folds split into train/val."""

    def __init__(
        self,
        n_splits: int = 5,
        fold_index: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        val_frac: float = 0.1,
    ):
        self.n_splits = n_splits
        self.fold_index = fold_index
        self.shuffle = shuffle
        self.seed = seed
        self.val_frac = val_frac

    def split(
        self,
        n_samples: int,
        y: np.ndarray = None,
        X: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if y is None:
            raise ValueError("StratifiedKFoldSplitter requires labels y.")

        indices = np.arange(n_samples)
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed
        )
        splits = list(skf.split(indices, y))
        train_val_idx, test_idx = splits[self.fold_index]

        # Carve out a val set from the train_val pool
        n_val = max(1, int(len(train_val_idx) * self.val_frac))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(train_val_idx))
        val_idx = train_val_idx[perm[:n_val]]
        train_idx = train_val_idx[perm[n_val:]]

        return train_idx, val_idx, test_idx
