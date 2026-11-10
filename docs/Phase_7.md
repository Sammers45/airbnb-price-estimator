# Phase 7: Ablation and significance tests (runs on your CPU, about 1-3 minutes)

`src/evaluate.py` answers one question with evidence: **does each input (text, photo) really improve the price estimate, or is the difference just noise?**

## What it does

1. **Leaderboard of every model** built so far, all scored on the same out-of-fold rows:
   LightGBM baseline (Phase 3), text only (Phase 4), fusion NN concat and attention (Phase 6).
2. **Ablation with a stacked LightGBM**: same folds, same outlier rule, four feature sets:

   | Feature set      | Extra inputs added to the table features                                                       |
   | ---------------- | ---------------------------------------------------------------------------------------------- |
   | `tab`            | none (same as the baseline, used as a control)                                                 |
   | `tab+text`       | text model's prediction + 8 principal components of the 768-d text vector                      |
   | `tab+image`      | photo model's prediction + 8 PCs of the 512-d photo vector + 12 prompt scores + has-photo flag |
   | `tab+text+image` | all of the above                                                                               |

   The PCA is **fit on the training fold only**, so validation rows never shape the features.

3. **Significance tests** for each question, on the **same listings** (paired):
   - **Paired bootstrap** (2,000 resamples of listings): change in MdAPE, MAE and within-10%, with a 95% interval and p-value.
   - **Paired t-test** on absolute log error (a second check).
   - **Holm correction** across all questions (we test about 8, so plain p < 0.05 would give false wins).
   - **Fold wins**: in how many of the 5 folds model B beat model A.
4. Picks the **best model** and saves its out-of-fold predictions for Phase 8 (intervals and human-review routing).

## Why a stacked LightGBM, when Phase 6 already has a fusion NN?

Phase 6 showed a small gain from fusion: networks stopped after 1-3 epochs and attention weights were near uniform. On ~5,000 rows of table data, gradient boosting is usually stronger than a neural network. Giving LightGBM the text and photo signals as **a few compact features** is the fair way to measure how much each input adds. Both approaches stay in the report; the significance test decides which is better.

## Important fix in this phase: `src/baseline.py`

`FeatureBuilder` picked up **every numeric column** as a table feature. After Phases 6-7 join other models' outputs (`base_log`, `text_pred_log`, `img_*`, `has_image`) onto the table, those columns leaked into the "table" features. In Phase 6 this meant the fusion network's table branch already saw the baseline prediction and photo prompt scores. The fix excludes them by name and prefix. **Rerun Phase 6 after this fix** so its numbers are clean.

## Run

```powershell
git checkout main
git pull
git checkout -b phase-7
# save src\evaluate.py, tests\test_evaluate.py, docs\PHASE_7.md
# edit src\baseline.py and tests\test_fusion.py; append the config section
pytest                         # 80 tests
python -m src.fusion train     # rerun Phase 6 with the fix (7-15 minutes)
python -m src.evaluate         # Phase 7 (1-3 minutes)
```

Needs these files from earlier phases (the script tells you which phase is missing):
`data/processed/listings.parquet`, `data/predictions/baseline_oof.parquet`, `text_oof.parquet`, `image_oof.parquet`, `fusion_concat_oof.parquet`, `fusion_attention_oof.parquet`.

## Outputs

| Path                                                           | Git? | Used by                                       |
| -------------------------------------------------------------- | ---- | --------------------------------------------- |
| `reports/evaluation.md`, `reports/evaluation_metrics.json`     | yes  | review, resume numbers                        |
| `reports/figures/eval_leaderboard.png`                         | yes  | README                                        |
| `data/predictions/all_models_oof.parquet`                      | no   | every model's prediction, one row per listing |
| `data/predictions/best_oof.parquet`, `best_model.json`         | no   | Phase 8 intervals                             |
| `models/stacked/stacked_lightgbm.txt`, `stacked_features.json` | no   | web app (Phase 9)                             |

## How to read the significance table

| Column   | Meaning                                                                                   |
| -------- | ----------------------------------------------------------------------------------------- |
| change   | MdAPE of B minus MdAPE of A, in percentage points (negative = B is better)                |
| 95% CI   | bootstrap interval; if it excludes 0, the change is real at that level                    |
| p (Holm) | adjusted p-value; below 0.05 = **better** / **worse**, else **no significant difference** |
| folds    | e.g. `5/5` = B beat A in every fold                                                       |

"No significant difference" is a valid result. For example, if the photo adds nothing on top of the text, saying so with a confidence interval is a stronger interview answer than an unproven gain.

## Interview notes

- **Why paired?** Both models predict the same listings, so comparing per-listing errors removes listing difficulty from the noise. The test is far more sensitive than comparing two averages.
- **Why bootstrap MdAPE?** The median has no simple standard-error formula; resampling listings gives one directly.
- **Stacking caveat:** the OOF inputs came from models trained on the other folds. This is standard practice and can make stacked scores slightly optimistic. It's stated in the report.
