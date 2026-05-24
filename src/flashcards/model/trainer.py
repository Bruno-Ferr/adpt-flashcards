"""Train the recall-prediction model.

Trains a logistic regression on the review history, saves a bundle
(model + scaler + feature names) to data/model.pkl, and logs the run
to the model_runs table.

Usage:
    python main.py --train
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from flashcards.db import queries
from flashcards.features.engineering import build_training_data

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "model.pkl"
MIN_SAMPLES = 30


def train(conn: sqlite3.Connection) -> tuple[float, int]:
    """Fit logistic regression, persist model, log run.

    Returns (val_accuracy, n_training_samples).
    Raises ValueError if there are fewer than MIN_SAMPLES training rows.
    """
    X, y = build_training_data(conn)

    if len(X) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(X)} training rows available (need ≥ {MIN_SAMPLES}). "
            "Run more review sessions first."
        )

    # Time-ordered split: last 20 % for validation
    split = max(1, int(len(X) * 0.8))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    val_acc = float(accuracy_score(y_val, model.predict(X_val_s)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": list(X.columns)},
        MODEL_PATH,
    )

    queries.insert_model_run(
        conn,
        n_samples=len(X_train),
        accuracy=val_acc,
        model_path=str(MODEL_PATH),
    )

    return val_acc, len(X_train)
