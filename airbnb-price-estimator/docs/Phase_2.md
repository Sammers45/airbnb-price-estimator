# Phase 2: Exploratory data analysis

Everything is in `src/eda.py` and uses **Polars**. It reads the Phase 1 output (`data/processed/listings.parquet`) and writes `reports/eda.md`. No new packages are needed.

## Key idea: within-market price

A Maui listing costs about 7× an Istanbul listing, so almost any feature "correlates" with price just by being more common in expensive markets. Most analyses therefore use

```
rel_log_price = log_price − median log_price of the listing's market
```

A feature only gets credit for what it says about price **inside** a market.

## What the report contains

| #   | Section                                      | Function                        | Why it matters                                                                                                                               |
| --- | -------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Target distribution and skew                 | `target_summary`                | Confirms predicting `log_price` is right                                                                                                     |
| 2   | Prices that are extreme for their market     | `suspicious_prices`             | Robust z-score (median and MAD) of log price above 3.5. These listings are reviewed by hand, not dropped.                                    |
| 3   | Price by market                              | `price_by` + box plot           | Market spread after USD conversion                                                                                                           |
| 4   | Price by room type and property type         | `price_by` + guests plot        | How capacity and room type drive price                                                                                                       |
| 5   | Numeric features vs within-market price      | `spearman_with_price`           | Ranks features before modelling                                                                                                              |
| 6   | Amenity price lift                           | `amenity_lift`                  | % price difference with vs without each amenity, within market                                                                               |
| 7   | Missing values                               | `missing_values`                | What Phase 3 must fill or flag                                                                                                               |
| 8   | Image coverage and text length               | `image_coverage`, `text_length` | Is a missing photo related to price? How long are descriptions for the Transformer?                                                          |
| 9   | Signal check: linear probes                  | `run_probes`                    | Out-of-fold R² of a ridge regression on market only, text embeddings and image embeddings. This is an early preview of the Phase 7 ablation. |
| 10  | Candidate feature: distance to market centre | `add_distance_to_centre`        | Great-circle km from the market's median location                                                                                            |

## Leakage note

The probes use the **Phase 1 folds**, and `RidgeCV` chooses its strength inside each training fold, so the R² values are honest out-of-fold numbers. The descriptive tables use all rows. They inform decisions but never fit anything a model later uses.

## Run

```powershell
git checkout main
git pull
git checkout -b phase-2

pytest                 # 35 tests (25 Phase 1 + 10 Phase 2)
python -m src.eda      # about 30 seconds; writes reports/eda.md
```

## Files in this phase

| File                | Status                                 |
| ------------------- | -------------------------------------- |
| `src/eda.py`        | new                                    |
| `tests/test_eda.py` | new                                    |
| `docs/PHASE_2.md`   | new                                    |
| `config.yaml`       | changed: new `eda:` section at the end |
| `README.md`         | changed: status table and quick start  |

## Send back for approval

`reports/eda.md`. From its numbers I'll write the findings and set the Phase 3 feature list: the outlier decision, which features to keep, and whether to add `dist_centre_km` and `has_reviews`.

## Commit

```powershell
git add src/eda.py tests/test_eda.py docs/PHASE_2.md config.yaml README.md reports/
git commit -m "feat: phase 2 EDA (within-market analysis, amenity lift, embedding probes)"
git push -u origin phase-2
```
