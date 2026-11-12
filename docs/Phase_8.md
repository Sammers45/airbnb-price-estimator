# Phase 8: Trust — price ranges, human review, explanations (CPU, about 1 minute)

Phase 7 showed every model is wrong by about 26% on a typical listing, and only ~20% of
predictions land within 10%. A pricing product can't hide that. It needs three things:

1. **A range, not just a number**: "$120, likely between $95 and $160", with a promise
   about how often the range is right.
2. **A rule for when a person should check**, so that the risky listings are reviewed and the
   rest are priced automatically.
3. **A reason**: which features pushed the price up or down.

Two scripts, one per job:

| Script             | What it does                                                   | Reads                                                            |
| ------------------ | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| `src/intervals.py` | conformal price ranges, coverage report, review routing        | `best_oof.parquet` (Phase 7)                                     |
| `src/explain.py`   | SHAP for LightGBM, Integrated Gradients for the fusion network | `models/baseline_*` (Phase 3), `models/fusion_concat/` (Phase 6) |

## 8a. Price ranges (`src/intervals.py`)

**Split conformal prediction**: on listings the model hasn't seen, measure how wrong it is,
then make the range just wide enough to contain the true price 80% (or 90%) of the time.
Its guarantee holds without assuming errors are normal.

Two versions are compared:

| Method     | Range                                                                                                                  | Weakness / strength                                         |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `constant` | same +/- on the log scale for everyone (e.g. ±25%)                                                                     | too wide for ordinary listings, too narrow for unusual ones |
| `adaptive` | ± scaled by a **difficulty model** (a small LightGBM that predicts how wrong the price model will be for this listing) | narrow when the listing is easy, wide when it's unusual     |

**Honest evaluation**: for each of the 5 folds, the difficulty model and the band sizes come
from the other 4 folds only. Within those 4 folds, 60% fits the difficulty model and 40% sizes
the band, so the band is sized on rows the difficulty model never saw. Coverage is then
measured on the held-out fold.

**What to look for in the report:**

- `coverage_pct` near the target (80 / 90) for both methods: the guarantee works.
- `coverage_hard_third`: `constant` usually falls well below target on hard listings;
  `adaptive` stays close. This is why adaptive ranges are worth it.
- Section 2: median actual error should rise with range width. A wide range really does mean
  "this one is risky".

## Human-review routing

A listing goes to a person when:

- its 80% range is wider than the widest 20% seen in calibration (`review_rate: 0.2`), **or**
- its market has fewer than 30 listings (`min_market_rows`). This catches the 6 listings with a
  blank market and the "Other" markets found in Phase 7.

The routing table answers the business question: **if a person checks X% of listings, how
accurate is the automatic price for the rest, and what share of the bad misses (>50% off) do
the reviewers catch?** These are the numbers for your resume.

## 8b. Explanations (`src/explain.py`)

**SHAP (Phase 3 LightGBM).** LightGBM computes exact TreeSHAP values itself
(`pred_contrib=True`), the same algorithm as the `shap` package, so no extra dependency. On the log
scale the values add up exactly to the prediction, so each one is a price multiplier:
`effect % = exp(value) - 1`. The report gives:

- global importance (average size of each feature's effect) and its direction,
- a beeswarm chart (`figures/shap_summary.png`),
- three listings explained one by one: budget, typical, premium.

`explain_listing()` returns the same explanation for one listing; the web app (Phase 9) uses it.

**Integrated Gradients (Captum, Phase 6 fusion network).** The network starts from the LightGBM
price and adds a correction. IG walks from an "average listing" to the real one and credits
each input with its share of that correction: table, description, photo. A missing photo is
masked, so it gets 0. The completeness check (attributions add up to the output change) is
printed in the report.

## Run

```powershell
git checkout main
git pull
git checkout -b phase-8
# save src\intervals.py, src\explain.py, tests\test_intervals.py, tests\test_explain.py, docs\PHASE_8.md
# append the config sections and the captum line to requirements.txt
pip install captum
pytest                         # 95 tests (80 + 9 + 6)
python -m src.intervals        # about 10 seconds
python -m src.explain          # about 30 seconds
```

## Outputs

| Path                                                                      | Git? | Used by                     |
| ------------------------------------------------------------------------- | ---- | --------------------------- |
| `reports/intervals.md`, `intervals_metrics.json`, `figures/intervals.png` | yes  | review, resume numbers      |
| `reports/explain.md`, `explain_metrics.json`, `figures/shap_summary.png`  | yes  | review, README              |
| `data/predictions/intervals_oof.parquet`                                  | no   | Phase 9 monitoring baseline |
| `models/intervals/` (difficulty model + band sizes + threshold)           | no   | web app (Phase 9)           |

## Interview notes

- **Why conformal?** It turns any point model into ranges with a coverage guarantee, using only
  held-out errors. No retraining of the price model, and no assumption that errors are normal.
- **Marginal vs conditional coverage:** conformal guarantees coverage _on average over
  listings_. Per-market coverage can drift, so the report shows it per market.
- **Why route on range width, not on the price?** Width is the model saying "I'm unsure". The
  report checks that this self-assessment matches real errors (section 2).
- **SHAP vs IG:** SHAP is exact for trees; IG is the gradient-based equivalent for neural
  networks. Both satisfy completeness (attributions add up to the prediction change).
