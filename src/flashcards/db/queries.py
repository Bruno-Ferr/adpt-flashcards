"""All SQL queries, wrapped as Python functions.

Keeping every query in one module means the rest of the codebase never writes
raw SQL inline. Higher layers (features, scheduler, cli) call these functions
and get back plain Python objects (``sqlite3.Row`` or simple values).
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def insert_card(
    conn: sqlite3.Connection,
    front: str,
    back: str,
    example: str | None = None,
    tags: str | None = None,
) -> int:
    """Insert a single card and return its new id."""
    cur = conn.execute(
        "INSERT INTO cards (front, back, example, tags) VALUES (?, ?, ?, ?);",
        (front, back, example, tags),
    )
    return int(cur.lastrowid)


def get_all_cards(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every card, oldest first."""
    return conn.execute(
        "SELECT id, front, back, example, tags, created_at "
        "FROM cards ORDER BY id;"
    ).fetchall()


def get_card(conn: sqlite3.Connection, card_id: int) -> sqlite3.Row | None:
    """Return one card by id, or ``None`` if it does not exist."""
    return conn.execute(
        "SELECT id, front, back, example, tags, created_at "
        "FROM cards WHERE id = ?;",
        (card_id,),
    ).fetchone()


def count_cards(conn: sqlite3.Connection) -> int:
    """Total number of cards in the deck."""
    return int(conn.execute("SELECT COUNT(*) FROM cards;").fetchone()[0])


def load_seed_deck(
    conn: sqlite3.Connection,
    csv_path: str | Path,
    skip_if_populated: bool = True,
) -> int:
    """Load a seed deck CSV into the ``cards`` table.

    The CSV must have a header row with columns: ``front, back, example, tags``.
    Returns the number of cards inserted.

    If ``skip_if_populated`` is True (default) and the deck already has cards,
    nothing is loaded and 0 is returned — so re-running the program does not
    create duplicates.
    """
    if skip_if_populated and count_cards(conn) > 0:
        return 0

    csv_path = Path(csv_path)
    inserted = 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            insert_card(
                conn,
                front=row["front"].strip(),
                back=row["back"].strip(),
                example=(row.get("example") or "").strip() or None,
                tags=(row.get("tags") or "").strip() or None,
            )
            inserted += 1
    conn.commit()
    return inserted


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #
def insert_review(
    conn: sqlite3.Connection,
    card_id: int,
    result: int,
    rating: int | None,
    response_time_ms: int | None,
    session_id: str,
) -> int:
    """Log one review event and return its new id.

    ``result`` is 0 (forgot) or 1 (recalled); ``rating`` is 1-4 or None.
    """
    cur = conn.execute(
        "INSERT INTO reviews "
        "(card_id, result, rating, response_time_ms, session_id) "
        "VALUES (?, ?, ?, ?, ?);",
        (card_id, result, rating, response_time_ms, session_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_reviews_for_card(
    conn: sqlite3.Connection, card_id: int
) -> list[sqlite3.Row]:
    """All reviews for one card, oldest first (for the forgetting curve)."""
    return conn.execute(
        "SELECT id, card_id, reviewed_at, result, rating, "
        "response_time_ms, session_id "
        "FROM reviews WHERE card_id = ? ORDER BY reviewed_at;",
        (card_id,),
    ).fetchall()


def get_all_reviews(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every review event, oldest first. The raw training set."""
    return conn.execute(
        "SELECT id, card_id, reviewed_at, result, rating, "
        "response_time_ms, session_id "
        "FROM reviews ORDER BY reviewed_at;"
    ).fetchall()


def count_reviews(conn: sqlite3.Connection) -> int:
    """Total number of review events logged so far."""
    return int(conn.execute("SELECT COUNT(*) FROM reviews;").fetchone()[0])


def get_last_review_per_card(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    """Map card_id -> its most recent review row.

    Useful for the scheduler's ``days_since_last_review`` without scanning the
    full history every time.
    """
    rows = conn.execute(
        "SELECT r.card_id, r.reviewed_at, r.result, r.rating, "
        "r.response_time_ms, r.session_id "
        "FROM reviews r "
        "JOIN (SELECT card_id, MAX(reviewed_at) AS mx "
        "      FROM reviews GROUP BY card_id) latest "
        "  ON r.card_id = latest.card_id AND r.reviewed_at = latest.mx;"
    ).fetchall()
    return {row["card_id"]: row for row in rows}


# --------------------------------------------------------------------------- #
# Model runs
# --------------------------------------------------------------------------- #
def insert_model_run(
    conn: sqlite3.Connection,
    n_samples: int,
    accuracy: float | None,
    model_path: str | None,
) -> int:
    """Record a training run and return its id."""
    cur = conn.execute(
        "INSERT INTO model_runs (n_samples, accuracy, model_path) "
        "VALUES (?, ?, ?);",
        (n_samples, accuracy, model_path),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_latest_model_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Most recent training run, or None if no model has been trained yet."""
    return conn.execute(
        "SELECT id, trained_at, n_samples, accuracy, model_path "
        "FROM model_runs ORDER BY trained_at DESC, id DESC LIMIT 1;"
    ).fetchone()