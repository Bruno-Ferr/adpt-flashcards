"""Database schema definitions and initialization.

Three tables, matching the project design:

- ``cards``       : every flashcard in the deck.
- ``reviews``     : one row per review event. This is the training data.
- ``model_runs``  : metadata for each model training run (reproducibility).

The schema is created idempotently via :func:`init_db`, so it is safe to call
on every program start. Foreign keys are enforced and indexes are added on the
columns the feature-engineering layer reads most often.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Default location of the SQLite database file. Resolves to
# ``<repo>/data/flashcards.db`` regardless of the current working directory.
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "flashcards.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    front       TEXT    NOT NULL,            -- German word or phrase
    back        TEXT    NOT NULL,            -- English translation
    example     TEXT,                        -- Example sentence in German
    tags        TEXT,                        -- e.g. noun, verb, separable-verb
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id          INTEGER NOT NULL,
    reviewed_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    result           INTEGER NOT NULL,       -- 0 = forgot, 1 = recalled
    rating           INTEGER,                -- 1=again 2=hard 3=good 4=easy
    response_time_ms INTEGER,                -- shown -> answer, milliseconds
    session_id       TEXT    NOT NULL,       -- groups reviews by session
    FOREIGN KEY (card_id) REFERENCES cards (id) ON DELETE CASCADE,
    CHECK (result IN (0, 1)),
    CHECK (rating IS NULL OR rating BETWEEN 1 AND 4)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    n_samples   INTEGER NOT NULL,            -- reviews used to train
    accuracy    REAL,                        -- validation accuracy
    model_path  TEXT                         -- path to serialized model
);

-- Feature engineering repeatedly pulls a card's review history in time order,
-- and the scheduler groups by session. These indexes keep those reads cheap.
CREATE INDEX IF NOT EXISTS idx_reviews_card_time
    ON reviews (card_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_reviews_session
    ON reviews (session_id);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with sensible defaults.

    - ``row_factory`` is set to :class:`sqlite3.Row` so columns are accessible
      by name (``row["front"]``) as well as by index.
    - Foreign key enforcement is turned on (off by default in SQLite).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create all tables and indexes if they do not already exist.

    Returns an open connection to the initialized database. Idempotent: running
    it against an existing database is a no-op.
    """
    conn = connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


if __name__ == "__main__":
    # Allow `python -m flashcards.db.schema` to bootstrap the database.
    conn = init_db()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    print(f"Initialized {DEFAULT_DB_PATH}")
    print("Tables:", ", ".join(r["name"] for r in tables))
    conn.close()