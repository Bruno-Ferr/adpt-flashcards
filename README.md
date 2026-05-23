# Adaptive Flashcards

A spaced-repetition flashcard system that learns *your* forgetting patterns
instead of using a fixed formula. German vocabulary, powered by your own
review data.

## Status

**Phase 1 complete** — data foundation and review loop.

- SQLite schema (`cards`, `reviews`, `model_runs`)
- 150-card German seed deck (tagged by type, gender, separable verbs)
- Interactive CLI review session that logs every answer as training data

Phases 2–4 (model, incremental learning, dashboard) are not built yet.

## Setup

```bash
pip install rich          # only runtime dependency so far
# (pytest for tests, optional)
```

Python 3.11+.

## Run a session

```bash
python main.py                # 20 cards (default)
python main.py --length 10    # shorter session
```

On first run it creates `data/flashcards.db` and loads the seed deck. Later
runs reuse the existing database and keep accumulating review history.

Each card: read the German front, press Enter to reveal the answer, then rate
your recall 1–4:

| Key | Meaning | Logged result |
|-----|---------|---------------|
| 1   | again   | forgot (0)    |
| 2   | hard    | recalled (1)  |
| 3   | good    | recalled (1)  |
| 4   | easy    | recalled (1)  |

The time from card-shown to reveal is recorded as `response_time_ms` — a
confidence signal for the model later. Press Ctrl-C to end early; whatever you
reviewed is still saved.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## Regenerate the seed deck

```bash
python scripts/gen_seed_deck.py
```

## Layout

```
src/flashcards/
  db/        schema + queries (all SQL lives here)
  cli/       session loop + post-session summary
  features/  (Phase 2) feature engineering
  model/     (Phase 2) train / predict / explain
  scheduler/ (Phase 2) model-driven card ranking
main.py      entry point
```

The card-selection step (`pick_cards` in `cli/session.py`) is the single seam
where the Phase 2 scheduler replaces random ordering — the review loop itself
won't change.