# Phase 3: Baseline (LightGBM on structured features) + MLflow

`src/baseline.py` trains a LightGBM model on structured features to predict `log1p(price)`. **Every later model has to beat it.** Shared metrics live in `src/metrics.py`, and every MLflow run goes through `mlflow_run()` in `src/utils.py`.

## How it works

| Step        | Function              | Detail                                                                                                                                                                                                                                                         |
| ----------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Split       | Phase 1 folds         | The same 5 folds for every model, so Phase 7 can run _paired_ tests                                                                                                                                                                                            |
| Outliers    | `drop_outliers`       | Training rows outside the training fold's 1st–99th percentile of log price are dropped. Validation rows are never dropped.                                                                                                                                     |
| Features    | `FeatureBuilder`      | Fit on the training fold only. Numeric columns, amenity flags, and categories (market, room type, property type, bed type, cancellation policy, neighbourhood). New: **`has_reviews`** and **`dist_centre_km`** (distance to the training-fold market centre). |
| Excluded    | `EXCLUDED`            | Target in any form (`price`, `log_price`, `price_local`), ids, and text and image inputs (they arrive in Phases 4–6)                                                                                                                                           |
| Train       | `fit_booster`         | Early stopping on 10% of the **training** fold, never the validation fold                                                                                                                                                                                      |
| Reference   | `naive_market_median` | Predicts the training-fold market median. Shows how much the model adds.                                                                                                                                                                                       |
| Final model | `train_final`         | All rows, number of trees = mean best iteration from CV. Saved for the web app.                                                                                                                                                                                |

## Metrics (`src/metrics.py`)

All metrics are computed in USD on out-of-fold predictions:

- **MAE**: mean absolute error in dollars
- **MdAPE**: median absolute percentage error
- **Within 10% / 20%**: share of listings priced within 10% or 20% of the actual price, the measures valuation teams use
- **R² (log)**: fit on the log scale

## Outputs

| Path                                                                                                  | Git? | Used by                             |
| ----------------------------------------------------------------------------------------------------- | ---- | ----------------------------------- |
| `reports/baseline.md`, `reports/baseline_metrics.json`, `reports/figures/baseline_pred_vs_actual.png` | yes  | review, resume numbers              |
| `data/predictions/baseline_oof.parquet`                                                               | no   | paired significance tests (Phase 7) |
| `models/baseline_lightgbm.txt`, `models/baseline_features.json`                                       | no   | web app (Phase 9)                   |
| `mlflow.db`, `mlruns/`                                                                                | no   | experiment tracking                 |

## Run

```powershell
git checkout main
git pull
git checkout -b phase-3

pip install -r requirements.txt     # adds lightgbm and mlflow
pytest                              # 45 tests (25 + 10 + 10)
python -m src.baseline              # about 1 minute
```

To see the run in MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000. You'll find the experiment **airbnb-price**, the run **baseline_lightgbm**, its parameters, the per-fold MAE chart and the artifacts.

## Files in this phase

| File                                                                             | Status                                                                               |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/baseline.py`, `src/metrics.py`, `tests/test_baseline.py`, `docs/PHASE_3.md` | new                                                                                  |
| `src/utils.py`                                                                   | changed: adds `mlflow_run()`                                                         |
| `config.yaml`                                                                    | changed: `paths.predictions`, `paths.models`, and `baseline:` and `mlflow:` sections |
| `requirements.txt`                                                               | changed: `lightgbm`, `mlflow`                                                        |
| `pyproject.toml`                                                                 | changed: ignores one MLflow-internal warning in tests                                |
| `.gitignore`                                                                     | changed: ignores `mlflow.db`                                                         |
| `README.md`                                                                      | changed: status table                                                                |

## Send back for approval

`reports/baseline.md`. These numbers become the baseline row of the Results table and the bar every later model has to beat.

## Commit

```powershell
git add src tests docs config.yaml requirements.txt pyproject.toml .gitignore README.md reports
git commit -m "feat: phase 3 LightGBM baseline with MLflow tracking"
git push -u origin phase-3
```
