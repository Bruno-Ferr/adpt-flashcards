"""Feature engineering for the recall-prediction model.

Two public functions:
- build_training_data(conn)   → (X: DataFrame, y: Series)
  One row per review event (skipping each card's very first review, which has
  no prior history to compute features from). Features reflect the card state
  *before* that review, so there is no data leakage.

- build_prediction_features(conn) → DataFrame indexed by card_id
  One row per card that has at least one review. Features reflect current state
  (history up to now), used by the scheduler to rank cards.

Feature set
-----------
n_reviews           : total prior reviews
recall_rate         : fraction recalled in prior reviews
last_result         : outcome of most recent review (0/1)
days_since_last     : days between last review and reference point
avg_response_ms     : mean response time across prior reviews
last_response_ms    : response time of most recent review
streak              : consecutive correct answers from the end of history
is_noun             : tag contains 'noun'
is_verb             : tag contains 'verb'
is_separable        : tag contains 'separable'
is_adjective        : tag contains 'adjective'
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from flashcards.db import queries

FEATURE_COLS = [
    "n_reviews",
    "recall_rate",
    "last_result",
    "days_since_last",
    "avg_response_ms",
    "last_response_ms",
    "streak",
    "is_noun",
    "is_verb",
    "is_separable",
    "is_adjective",
]


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def _days_between(earlier: str, later: str) -> float:
    return (_parse_ts(later) - _parse_ts(earlier)).total_seconds() / 86_400


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _features_from_history(history: list[sqlite3.Row], card: sqlite3.Row, reference_ts: str) -> dict:
    n = len(history)
    recalled = [r["result"] for r in history]
    times = [r["response_time_ms"] for r in history if r["response_time_ms"] is not None]

    streak = 0
    for r in reversed(recalled):
        if r == 1:
            streak += 1
        else:
            break

    tags = card["tags"] or ""

    return {
        "n_reviews": n,
        "recall_rate": sum(recalled) / n,
        "last_result": history[-1]["result"],
        "days_since_last": _days_between(history[-1]["reviewed_at"], reference_ts),
        "avg_response_ms": sum(times) / len(times) if times else 0.0,
        "last_response_ms": float(history[-1]["response_time_ms"] or 0),
        "streak": float(streak),
        "is_noun": float("noun" in tags),
        "is_verb": float("verb" in tags),
        "is_separable": float("separable" in tags),
        "is_adjective": float("adjective" in tags),
    }


def _group_by_card(reviews: list[sqlite3.Row]) -> dict[int, list[sqlite3.Row]]:
    by_card: dict[int, list] = defaultdict(list)
    for r in reviews:
        by_card[r["card_id"]].append(r)
    return by_card


def build_training_data(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) training set from review history.

    Reviews are already sorted oldest-first by get_all_reviews. For each card
    we iterate its history and produce one training row per review after the
    first, using only the reviews that preceded it as features.
    """
    all_cards = {c["id"]: c for c in queries.get_all_cards(conn)}
    by_card = _group_by_card(queries.get_all_reviews(conn))

    feature_rows: list[dict] = []
    targets: list[int] = []

    for card_id, reviews in by_card.items():
        card = all_cards.get(card_id)
        if card is None:
            continue
        for i in range(1, len(reviews)):
            history = reviews[:i]
            target = reviews[i]
            row = _features_from_history(history, card, reference_ts=target["reviewed_at"])
            feature_rows.append(row)
            targets.append(int(target["result"]))

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
    y = pd.Series(targets, name="result")
    return X, y


def build_prediction_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build current-state features for every card with at least one review.

    Returns a DataFrame indexed by card_id.
    """
    all_cards = {c["id"]: c for c in queries.get_all_cards(conn)}
    by_card = _group_by_card(queries.get_all_reviews(conn))

    now = _now_str()
    rows: list[dict] = []

    for card_id, reviews in by_card.items():
        card = all_cards.get(card_id)
        if card is None or not reviews:
            continue
        row = _features_from_history(reviews, card, reference_ts=now)
        row["card_id"] = card_id
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS)

    df = pd.DataFrame(rows).set_index("card_id")
    return df[FEATURE_COLS]
