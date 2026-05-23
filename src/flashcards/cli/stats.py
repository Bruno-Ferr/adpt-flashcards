"""Post-session summary display.

Renders a compact recap of a completed :class:`SessionResult`: how many cards
were reviewed, accuracy, average response time, and a per-card breakdown.
Kept separate from the loop so the display can grow (Phase 4 dashboard) without
touching the session logic.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from flashcards.cli.session import RATING_MAP, SessionResult


def show_summary(result: SessionResult, console: Console | None = None) -> None:
    """Print a summary table for a finished session."""
    console = console or Console()

    if result.n_reviewed == 0:
        console.print("[yellow]No cards reviewed this session.[/]")
        return

    console.rule("[bold]Session summary[/]")

    # headline stats
    headline = Table.grid(padding=(0, 3))
    headline.add_row(
        f"[bold]{result.n_reviewed}[/] reviewed",
        f"[bold]{result.n_recalled}[/] recalled",
        f"[bold]{result.accuracy:.0%}[/] accuracy",
        f"[bold]{result.avg_response_ms/1000:.1f}s[/] avg",
    )
    console.print(headline)
    console.print()

    # per-card breakdown
    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Card", style="cyan", no_wrap=True)
    table.add_column("Rating")
    table.add_column("Result", justify="center")
    table.add_column("Time", justify="right")

    for r in result.reviewed:
        label = RATING_MAP[r["rating"]][0]
        mark = "[green]✓[/]" if r["result"] == 1 else "[red]✗[/]"
        table.add_row(
            r["front"],
            label,
            mark,
            f"{r['response_time_ms']/1000:.1f}s",
        )

    console.print(table)
    console.print(f"\n[dim]session {result.session_id}[/]")