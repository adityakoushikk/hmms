"""Isolation Forest wrapper for the anomaly detection pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    """Thin wrapper around sklearn IsolationForest with a unified interface.

    anomaly_scores() returns higher values for more anomalous providers
    (negates score_samples which gives log-density — lower = more anomalous).
    """

    is_neural: bool = False

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        max_samples: str | int | float = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        evaluation_set: str = "all",
        **kwargs,
    ):
        self.evaluation_set = evaluation_set
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """Fit on X (labels ignored — unsupervised)."""
        self._model.fit(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Higher score → more anomalous."""
        return -self._model.score_samples(X)
