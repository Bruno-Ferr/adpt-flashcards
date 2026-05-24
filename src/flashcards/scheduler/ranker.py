"""Model-driven card ranking.

rank_cards() is the Phase 2 replacement for the random shuffle in pick_cards().
It scores every card with a review history as P(forget) = 1 - P(recall) and
returns the top-n cards most at risk of being forgotten.

Cards with no review history are appended after the ranked list so the session
always fills up to n cards even when the model can't score some of them.
"""
from __future__ import annotations

import sqlite3

from flashcards.db import queries
from flashcards.features.engineering import build_prediction_features
from flashcards.model.predictor import Predictor


def rank_cards(conn: sqlite3.Connection, n: int) -> list[sqlite3.Row]:
    """Return up to n cards ranked by descending P(forget).

    Cards the model can score (have review history) come first, ordered by
    how likely they are to be forgotten right now. Any remaining slots are
    filled with unreviewed cards in insertion order.
    """
    import flashcards.model.predictor as _pred_mod
    predictor = Predictor.load(_pred_mod.MODEL_PATH)
    if predictor is None:
        raise RuntimeError(
            "No trained model found. Run `python main.py --train` first."
        )

    features_df = build_prediction_features(conn)

    if not features_df.empty:
        p_recall = predictor.predict_recall_proba(features_df)
        features_df = features_df.copy()
        features_df["p_forget"] = 1.0 - p_recall
        ranked_ids = list(
            features_df.sort_values("p_forget", ascending=False).index
        )
    else:
        ranked_ids = []

    ranked_id_set = set(ranked_ids)
    all_cards = queries.get_all_cards(conn)
    unreviewed = [c for c in all_cards if c["id"] not in ranked_id_set]

    ordered_ids = ranked_ids[:n]
    if len(ordered_ids) < n:
        needed = n - len(ordered_ids)
        ordered_ids += [c["id"] for c in unreviewed[:needed]]

    cards_by_id = {c["id"]: c for c in all_cards}
    return [cards_by_id[cid] for cid in ordered_ids if cid in cards_by_id]
