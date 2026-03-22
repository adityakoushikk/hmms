"""Logistic Regression wrapper for the anomaly detection pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


class LogisticRegressionModel:
    """Supervised model; anomaly_scores returns P(fraud).

    Expects stratified_kfold splitter so scores are evaluated on held-out test data.
    """

    is_neural: bool = False

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        evaluation_set: str = "test",
        **kwargs,
    ):
        self.evaluation_set = evaluation_set
        self._model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) for each provider."""
        return self._model.predict_proba(X)[:, 1]

    def extra_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """AUROC and average precision (only meaningful with both classes present)."""
        if len(np.unique(y)) < 2:
            return {"auroc": float("nan"), "avg_precision": float("nan")}
        scores = self.anomaly_scores(X)
        return {
            "auroc": round(float(roc_auc_score(y, scores)), 4),
            "avg_precision": round(float(average_precision_score(y, scores)), 4),
        }
