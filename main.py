"""Entry point — run a review session.

Usage:
    python main.py                # 20-card session (default)
    python main.py --length 10    # shorter session
    python main.py --seed data/seed_deck.csv

On first run it initializes the database and loads the seed deck. On later runs
those steps are no-ops (the deck loader skips if cards already exist).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rich.console import Console

from flashcards.cli.session import DEFAULT_SESSION_LENGTH, run_session
from flashcards.cli.stats import show_summary
from flashcards.db import queries, schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive flashcard review session")
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_SESSION_LENGTH,
        help=f"number of cards to review (default {DEFAULT_SESSION_LENGTH})",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "seed_deck.csv"),
        help="path to the seed deck CSV (loaded only if the deck is empty)",
    )
    args = parser.parse_args(argv)

    console = Console()
    conn = schema.init_db()

    loaded = queries.load_seed_deck(conn, args.seed)
    if loaded:
        console.print(f"[green]Loaded {loaded} cards from seed deck.[/]\n")

    result = run_session(conn, length=args.length, console=console)
    show_summary(result, console=console)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())