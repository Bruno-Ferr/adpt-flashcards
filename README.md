# Adaptive Flashcard System

An adaptive spaced-repetition system for German vocabulary. Instead of reviewing cards on a fixed interval, the system predicts *when I'm likely to forget a specific card* and prioritizes reviews accordingly — treating scheduling as a learned problem rather than a fixed formula.

## Why this exists

Most flashcard apps (Anki included) use fixed or lightly-tuned spaced-repetition formulas (SM-2 and variants). They work, but they treat every learner and every card the same way. I wanted to know: could a model trained on my *own* review history do better than a generic formula, and — more importantly — could I understand *why* it makes the calls it makes?

This project prioritizes the model and the reasoning behind it over the UI. There's no polished frontend here on purpose; the interesting part is the decision layer underneath.

## How it works

**1. Recall prediction**
For each card, the system predicts the probability I'll recall it correctly on the next review, using features like:
- Time since last review
- Historical accuracy on that card
- Card difficulty (estimated from aggregate performance)
- Review streak / lapse count

I started with **logistic regression** as a baseline — simple, interpretable, fast to iterate on. Once the feature set stabilized, I moved to **XGBoost**, which captured non-linear interactions the logistic model missed (e.g. the combined effect of a long gap *and* a recent lapse is worse than either alone) and meaningfully improved recall prediction accuracy.

**2. Model interpretation**
Moving to XGBoost trades off interpretability, so I used **SHAP** to keep the model's reasoning visible — confirming which features actually drive forgetting for a given card, and catching cases where the model was leaning on a feature I didn't expect (e.g. overweighting recency vs. lapse count). This mattered to me more than squeezing out extra accuracy: a scheduler I can't explain isn't one I trust with my own study time.

**3. Scheduling**
Recall prediction alone isn't a scheduler — it just tells you what's at risk. On top of it, I built an **ε-greedy scheduler**: most of the time it prioritizes cards with the lowest predicted recall probability (exploit), but with probability ε it surfaces an under-reviewed or new card instead (explore), so the system doesn't get stuck only reinforcing cards it's already confident about.

**4. Storage**
Review history (timestamps, outcomes, per-card stats) is stored in **SQLite** — enough for a single-user system like this, and it keeps the whole thing runnable locally without extra infrastructure.

## Architecture

```
[Review session] → [SQLite: log outcome]
                          ↓
              [Feature engineering: recency, accuracy, lapses, difficulty]
                          ↓
              [XGBoost: predict recall probability per card]
                          ↓
              [ε-greedy scheduler: pick next card(s)]
                          ↓
                  [Review session]
```

## Results

*(Fill in with your actual numbers before publishing — e.g. logistic regression baseline accuracy/AUC vs. XGBoost, how much SHAP-driven feature pruning changed performance, or a before/after on retention rate over N weeks of real use.)*

## What I'd do differently / next steps

- [ ] Compare against a standard SM-2 baseline directly, not just logistic regression, to make the "does ML help here" case more rigorous
- [ ] Try a bandit approach with a real reward signal (e.g. Thompson sampling) instead of fixed ε, and see if it adapts faster
- [ ] Expand the feature set with semantic difficulty (e.g. word frequency, cognate similarity to Portuguese/English) rather than only behavioral features

## Tech stack

Python · scikit-learn · XGBoost · SHAP · SQLite

## Running it locally

*(Add setup instructions once the repo structure is finalized — dependencies, how to seed the DB, how to run a review session.)*

---

Part of a broader self-tracking/ML portfolio — see [Personal State & Activity Recommender](link) for a related project using real personal data to reduce decision paralysis.
