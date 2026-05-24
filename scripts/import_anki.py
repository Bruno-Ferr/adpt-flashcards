"""Import German vocabulary and review history from Anki.

Replaces all cards and reviews in the flashcards DB with data from the
Anki 'Alemão' deck. Run this once before training the model.

NOTE: Re-running wipes any reviews done inside this app since the last
import. Run it only when you want to refresh from Anki wholesale.

Usage:
    python scripts/import_anki.py
    python scripts/import_anki.py --anki-db /path/to/collection.anki2
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flashcards.db import queries, schema

ANKI_DB_DEFAULT = Path.home() / ".local/share/Anki2/Usuário 1/collection.anki2"
GERMAN_DECK_ID = 1704316214837


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&nbsp;", " ").strip()


def _parse_cloze(fld: str, ord_: int) -> tuple[str, str]:
    """Return (front_with_blank, back_word) for a cloze card.

    ord_ is 0-indexed (Anki card.ord), so c1 = ord 0, c2 = ord 1, etc.
    Other cloze slots in the same note are revealed in the front.
    """
    target_n = ord_ + 1
    pattern = re.compile(r"\{\{c(\d+)::([^}:]+)(?:::[^}]*)?\}\}")

    match = pattern.search(fld, re.IGNORECASE.value)  # find target first
    target_match = None
    for m in pattern.finditer(fld):
        if int(m.group(1)) == target_n:
            target_match = m
            break

    if target_match is None:
        return _strip_html(fld), ""

    back = target_match.group(2).strip()

    def _replace(m: re.Match) -> str:
        return "___" if int(m.group(1)) == target_n else m.group(2)

    front = _strip_html(pattern.sub(_replace, fld))
    return front, back


def import_anki(anki_db: Path, our_db: Path) -> None:
    anki = sqlite3.connect(anki_db)
    anki.row_factory = sqlite3.Row

    conn = schema.init_db(our_db)

    conn.execute("DELETE FROM reviews;")
    conn.execute("DELETE FROM cards;")
    conn.commit()
    print("Cleared existing cards and reviews.")

    card_rows = anki.execute(
        """
        SELECT c.id AS cid, c.ord, n.flds, n.tags
        FROM cards c
        JOIN notes n ON c.nid = n.id
        WHERE c.did = ?
        ORDER BY c.id;
        """,
        (GERMAN_DECK_ID,),
    ).fetchall()

    anki_cid_to_id: dict[int, int] = {}
    skipped = 0

    for row in card_rows:
        fields = row["flds"].split("\x1f")
        fld0 = fields[0] if fields else ""
        fld1 = fields[1] if len(fields) > 1 else ""

        front, back = _parse_cloze(fld0, row["ord"])
        if not front or not back:
            skipped += 1
            continue

        example = _strip_html(fld1) if fld1 else None
        tags = row["tags"].strip() or None

        card_id = queries.insert_card(conn, front=front, back=back, example=example, tags=tags)
        anki_cid_to_id[row["cid"]] = card_id

    conn.commit()
    print(f"Imported {len(anki_cid_to_id)} cards ({skipped} skipped — no cloze match).")

    cids = list(anki_cid_to_id)
    placeholders = ",".join("?" * len(cids))
    revlog = anki.execute(
        f"SELECT id, cid, ease, time FROM revlog WHERE cid IN ({placeholders}) ORDER BY id;",
        cids,
    ).fetchall()

    # Group same-day reviews into one synthetic session
    session_map: dict[str, str] = {}
    review_count = 0

    for rev in revlog:
        our_id = anki_cid_to_id.get(rev["cid"])
        if our_id is None:
            continue
        ease = rev["ease"]
        if ease not in (1, 2, 3, 4):
            continue

        result = 0 if ease == 1 else 1
        response_ms = max(0, int(rev["time"]))
        ts = datetime.fromtimestamp(rev["id"] / 1000, tz=timezone.utc)
        day_key = ts.strftime("%Y-%m-%d")
        session_id = session_map.setdefault(day_key, str(uuid.uuid4()))

        conn.execute(
            "INSERT INTO reviews "
            "(card_id, reviewed_at, result, rating, response_time_ms, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (our_id, ts.strftime("%Y-%m-%d %H:%M:%S"), result, ease, response_ms, session_id),
        )
        review_count += 1

    conn.commit()
    print(f"Imported {review_count} reviews across {len(session_map)} sessions.")
    anki.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Anki German deck into flashcards DB")
    parser.add_argument("--anki-db", default=str(ANKI_DB_DEFAULT), help="Path to collection.anki2")
    parser.add_argument("--db", default=None, help="Override flashcards DB path")
    args = parser.parse_args()

    anki_path = Path(args.anki_db)
    our_path = schema.DEFAULT_DB_PATH if args.db is None else Path(args.db)

    if not anki_path.exists():
        print(f"Anki DB not found: {anki_path}", file=sys.stderr)
        sys.exit(1)

    import_anki(anki_path, our_path)


if __name__ == "__main__":
    main()
