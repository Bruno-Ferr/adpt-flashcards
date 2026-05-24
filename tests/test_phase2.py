"""Phase 2 tests: feature engineering, model, and scheduler."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from flashcards.cli.session import pick_cards
from flashcards.db import queries, schema
from flashcards.features.engineering import (
    FEATURE_COLS,
    build_prediction_features,
    build_training_data,
)
from flashcards.model.predictor import Predictor
from flashcards.model.trainer import MIN_SAMPLES, train
from flashcards.scheduler.ranker import rank_cards


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def conn(tmp_path):
    db = tmp_path / "test.db"
    c = schema.init_db(db)
    yield c
    c.close()


def _seed(conn: sqlite3.Connection, n_cards: int = 5, reviews_per_card: int = 4) -> None:
    """Insert n_cards cards with reviews_per_card review events each."""
    import uuid
    from datetime import datetime, timedelta, timezone

    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    session_id = str(uuid.uuid4())

    for i in range(n_cards):
        cid = queries.insert_card(conn, front=f"Wort{i}", back=f"word{i}", tags="noun")
        for j in range(reviews_per_card):
            ts = base_ts + timedelta(days=j * 3)
            # Alternate outcomes so the training set has both classes
            result = j % 2
            conn.execute(
                "INSERT INTO reviews (card_id, reviewed_at, result, rating, response_time_ms, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (cid, ts.strftime("%Y-%m-%d %H:%M:%S"), result, 3, 1500, session_id),
            )
    conn.commit()


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_build_training_data_shape(conn):
    _seed(conn, n_cards=5, reviews_per_card=4)
    X, y = build_training_data(conn)
    # 5 cards × 3 rows each (skip first review per card)
    assert len(X) == 15
    assert list(X.columns) == FEATURE_COLS
    assert len(y) == 15


def test_build_training_data_no_leakage(conn):
    _seed(conn, n_cards=3, reviews_per_card=4)
    X, y = build_training_data(conn)
    # n_reviews in every row must be ≥ 1 (always have prior history)
    assert (X["n_reviews"] >= 1).all()


def test_build_prediction_features_indexed_by_card_id(conn):
    _seed(conn, n_cards=4, reviews_per_card=2)
    df = build_prediction_features(conn)
    assert df.index.name == "card_id"
    assert len(df) == 4
    assert list(df.columns) == FEATURE_COLS


def test_no_reviews_returns_empty(conn):
    queries.insert_card(conn, front="Haus", back="house")
    df = build_prediction_features(conn)
    assert df.empty


# --------------------------------------------------------------------------- #
# Model training
# --------------------------------------------------------------------------- #
def test_train_saves_model(conn, tmp_path):
    _seed(conn, n_cards=10, reviews_per_card=6)
    import flashcards.model.trainer as trainer_mod

    original = trainer_mod.MODEL_PATH
    trainer_mod.MODEL_PATH = tmp_path / "model.pkl"
    try:
        val_acc, n = train(conn)
        assert 0.0 <= val_acc <= 1.0
        assert n > 0
        assert trainer_mod.MODEL_PATH.exists()
    finally:
        trainer_mod.MODEL_PATH = original


def test_train_raises_when_too_few_samples(conn):
    _seed(conn, n_cards=2, reviews_per_card=2)
    with pytest.raises(ValueError, match="training rows"):
        train(conn)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
def test_predictor_output_range(conn, tmp_path):
    _seed(conn, n_cards=10, reviews_per_card=6)
    import flashcards.model.trainer as trainer_mod

    model_path = tmp_path / "model.pkl"
    trainer_mod.MODEL_PATH = model_path
    try:
        train(conn)
    finally:
        trainer_mod.MODEL_PATH = trainer_mod.MODEL_PATH  # restore not needed

    predictor = Predictor.load(model_path)
    assert predictor is not None

    features_df = build_prediction_features(conn)
    proba = predictor.predict_recall_proba(features_df)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predictor_load_missing_returns_none(tmp_path):
    result = Predictor.load(tmp_path / "nonexistent.pkl")
    assert result is None


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def test_rank_cards_returns_n(conn, tmp_path):
    _seed(conn, n_cards=10, reviews_per_card=6)
    import flashcards.model.trainer as trainer_mod
    import flashcards.model.predictor as predictor_mod
    import flashcards.scheduler.ranker as ranker_mod

    model_path = tmp_path / "model.pkl"
    trainer_mod.MODEL_PATH = model_path
    predictor_mod.MODEL_PATH = model_path
    try:
        train(conn)
        cards = rank_cards(conn, 5)
        assert len(cards) == 5
    finally:
        trainer_mod.MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model.pkl"
        predictor_mod.MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model.pkl"


def test_pick_cards_falls_back_to_random_without_model(conn):
    import random
    _seed(conn, n_cards=5, reviews_per_card=2)
    # MODEL_PATH doesn't exist in test environment → falls back to random
    cards = pick_cards(conn, 3, rng=random.Random(0))
    assert len(cards) == 3
