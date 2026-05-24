"""Interactive review session — the core daily loop.

Flow for each card:
  1. Show the front (German). Start the response timer.
  2. User presses Enter to reveal the back (English) + example. The timer
     stops here — response_time_ms measures the recall attempt, not the
     time spent choosing a rating afterward.
  3. User rates recall 1-4 (again / hard / good / easy).
  4. The event is logged to ``reviews``: result (0/1), rating (1-4),
     response_time_ms, and the session_id.

Card selection is deliberately isolated in ``pick_cards`` so that Phase 2 can
replace random ordering with the model-driven scheduler without touching the
loop itself. Rating -> result mapping: rating 1 ("again") = forgot (0);
ratings 2-4 = recalled (1).
"""

from __future__ import annotations

import random
import sqlite3
import time
import uuid
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from flashcards.db import queries

# rating value -> (label, result). result is the binary recall label.
RATING_MAP: dict[int, tuple[str, int]] = {
    1: ("again", 0),
    2: ("hard", 1),
    3: ("good", 1),
    4: ("easy", 1),
}

DEFAULT_SESSION_LENGTH = 20


@dataclass
class SessionResult:
    """Summary of one completed session, handed to the stats display."""

    session_id: str
    reviewed: list[dict]  # one dict per card reviewed

    @property
    def n_reviewed(self) -> int:
        return len(self.reviewed)

    @property
    def n_recalled(self) -> int:
        return sum(r["result"] for r in self.reviewed)

    @property
    def accuracy(self) -> float:
        return self.n_recalled / self.n_reviewed if self.reviewed else 0.0

    @property
    def avg_response_ms(self) -> float:
        if not self.reviewed:
            return 0.0
        return sum(r["response_time_ms"] for r in self.reviewed) / len(self.reviewed)


def pick_cards(
    conn: sqlite3.Connection, n: int, *, rng: random.Random | None = None
) -> list[sqlite3.Row]:
    """Choose up to ``n`` cards for this session.

    Uses the trained model via scheduler.ranker when a model file exists.
    Falls back to random selection otherwise — run `python main.py --train`
    to build the model after accumulating review history.
    """
    from flashcards.model.predictor import MODEL_PATH

    if MODEL_PATH.exists():
        from flashcards.scheduler.ranker import rank_cards
        return rank_cards(conn, n)

    rng = rng or random.Random()
    cards = queries.get_all_cards(conn)
    rng.shuffle(cards)
    return cards[:n]


def _prompt_rating(console: Console) -> int:
    """Ask for a 1-4 rating, re-prompting until valid. Returns the int."""
    choice = Prompt.ask(
        "  [bold]Rate recall[/]  "
        "[dim]\\[1] again  \\[2] hard  \\[3] good  \\[4] easy[/]",
        choices=["1", "2", "3", "4"],
        show_choices=False,
    )
    return int(choice)


def review_one(
    conn: sqlite3.Connection,
    card: sqlite3.Row,
    session_id: str,
    console: Console,
) -> dict:
    """Run the show -> reveal -> rate cycle for a single card and log it.

    Returns a dict describing what was logged (used for the session summary).
    """
    # --- show front ---
    front = Text(card["front"], style="bold cyan", justify="center")
    console.print(Panel(front, title="[dim]recall this[/]", border_style="cyan"))

    # Timer measures the recall attempt: from card shown to the moment the
    # user reveals the answer. That latency is the confidence signal the
    # feature layer wants ("slower = less confident"). The time spent picking
    # a 1-4 rating afterwards is motor reaction, not recall, so it's excluded.
    start = time.monotonic()
    Prompt.ask("  [dim]press Enter to reveal[/]", default="", show_default=False)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # --- reveal back + example ---
    back = Text(card["back"], style="bold green", justify="center")
    body = back
    if card["example"]:
        body = Text.assemble(
            back, "\n\n", Text(card["example"], style="italic dim", justify="center")
        )
    console.print(Panel(body, title="[dim]answer[/]", border_style="green"))

    rating = _prompt_rating(console)

    label, result = RATING_MAP[rating]

    queries.insert_review(
        conn,
        card_id=card["id"],
        result=result,
        rating=rating,
        response_time_ms=elapsed_ms,
        session_id=session_id,
    )

    mark = "[green]✓[/]" if result == 1 else "[red]✗[/]"
    console.print(
        f"  {mark} logged: [bold]{label}[/] · {elapsed_ms/1000:.1f}s\n"
    )

    return {
        "card_id": card["id"],
        "front": card["front"],
        "result": result,
        "rating": rating,
        "response_time_ms": elapsed_ms,
    }


def run_session(
    conn: sqlite3.Connection,
    length: int = DEFAULT_SESSION_LENGTH,
    *,
    console: Console | None = None,
) -> SessionResult:
    """Run a full review session of up to ``length`` cards.

    Ends when ``length`` cards are reviewed or the user quits (Ctrl-C / 'q'
    at the rating prompt is handled by the caller / KeyboardInterrupt).
    """
    console = console or Console()
    session_id = str(uuid.uuid4())

    if queries.count_cards(conn) == 0:
        console.print("[red]No cards in the deck. Load the seed deck first.[/]")
        return SessionResult(session_id=session_id, reviewed=[])

    cards = pick_cards(conn, length)
    console.print(
        Panel(
            f"[bold]Review session[/] · {len(cards)} cards\n"
            f"[dim]session {session_id[:8]}[/]",
            border_style="magenta",
        )
    )

    reviewed: list[dict] = []
    try:
        for i, card in enumerate(cards, 1):
            console.rule(f"[dim]{i}/{len(cards)}[/]")
            reviewed.append(review_one(conn, card, session_id, console))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Session ended early.[/]")

    return SessionResult(session_id=session_id, reviewed=reviewed)