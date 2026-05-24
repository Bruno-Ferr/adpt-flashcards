"""Load the trained model and predict P(recall) for feature rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "model.pkl"


class Predictor:
    def __init__(self, bundle: dict) -> None:
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._features: list[str] = bundle["features"]

    def predict_recall_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(recall=1) for each row in *features*."""
        X = features[self._features]
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Predictor | None":
        """Return a Predictor if the model file exists, else None."""
        if not path.exists():
            return None
        import joblib
        return cls(joblib.load(path))
