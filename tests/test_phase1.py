"""Phase 1 tests: schema integrity, query behavior, and session logic.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
Uses an in-memory / temp database so nothing touches the real deck.
"""

from __future__ import annotations

import sqlite3

import pytest

from flashcards.cli.session import RATING_MAP, SessionResult, pick_cards, review_one
from flashcards.db import queries, schema


@pytest.fixture()
def conn(tmp_path):
    """A fresh, initialized database in a temp file per test."""
    db = tmp_path / "test.db"
    c = schema.init_db(db)
    yield c
    c.close()


@pytest.fixture()
def seeded(conn, tmp_path):
    """A connection with a tiny 3-card deck loaded."""
    csv_path = tmp_path / "deck.csv"
    csv_path.write_text(
        "front,back,example,tags\n"
        "Haus,house,Das Haus.,noun\n"
        "gehen,to go,Ich gehe.,verb\n"
        "gut,good,Sehr gut.,adjective\n",
        encoding="utf-8",
    )
    queries.load_seed_deck(conn, csv_path)
    return conn


# --- schema -------------------------------------------------------------- #
def test_tables_exist(conn):
    names = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    }
    assert {"cards", "reviews", "model_runs"} <= names


def test_foreign_key_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        queries.insert_review(conn, 9999, 1, 3, 100, "s")


def test_result_check_constraint(conn):
    cid = queries.insert_card(conn, "a", "b")
    with pytest.raises(sqlite3.IntegrityError):
        # result must be 0 or 1
        conn.execute(
            "INSERT INTO reviews (card_id, result, session_id) VALUES (?, 5, 's');",
            (cid,),
        )


# --- queries ------------------------------------------------------------- #
def test_seed_load_idempotent(seeded, tmp_path):
    assert queries.count_cards(seeded) == 3
    # loading again should insert nothing
    again = queries.load_seed_deck(seeded, tmp_path / "deck.csv")
    assert again == 0
    assert queries.count_cards(seeded) == 3


def test_review_roundtrip(seeded):
    queries.insert_review(seeded, 1, 1, 3, 1500, "sess-1")
    rows = queries.get_reviews_for_card(seeded, 1)
    assert len(rows) == 1
    assert rows[0]["result"] == 1
    assert rows[0]["response_time_ms"] == 1500


def test_last_review_per_card(seeded):
    queries.insert_review(seeded, 1, 0, 1, 100, "s")
    queries.insert_review(seeded, 1, 1, 4, 200, "s")
    last = queries.get_last_review_per_card(seeded)
    assert last[1]["result"] == 1  # most recent wins


# --- session logic ------------------------------------------------------- #
def test_rating_map_consistency():
    # rating 1 = forgot, 2-4 = recalled
    assert RATING_MAP[1][1] == 0
    assert all(RATING_MAP[r][1] == 1 for r in (2, 3, 4))


def test_pick_cards_respects_length(seeded):
    import random

    picked = pick_cards(seeded, 2, rng=random.Random(0))
    assert len(picked) == 2


def test_pick_cards_caps_at_deck_size(seeded):
    picked = pick_cards(seeded, 100)
    assert len(picked) == 3  # only 3 cards exist


def test_session_result_metrics():
    res = SessionResult(
        session_id="x",
        reviewed=[
            {"card_id": 1, "front": "a", "result": 1, "rating": 3, "response_time_ms": 1000},
            {"card_id": 2, "front": "b", "result": 0, "rating": 1, "response_time_ms": 2000},
        ],
    )
    assert res.n_reviewed == 2
    assert res.n_recalled == 1
    assert res.accuracy == 0.5
    assert res.avg_response_ms == 1500