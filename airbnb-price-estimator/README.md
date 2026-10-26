# Airbnb Price Estimator

Predicts an Airbnb listing's nightly price from its structured fields, description text and photo. The models are a LightGBM baseline, a fine-tuned Transformer for text, CLIP for the photo, and a fusion neural network that combines them. It adds explanations (SHAP and Integrated Gradients), prediction intervals that send uncertain listings to human review, and a small web app with monitoring.

> Every number in this README comes from a script that was actually run. Until then it says `TBD`.

## Plan: one small module per phase

| Phase | What                                                            | Files added                                   | Status        |
| ----- | --------------------------------------------------------------- | --------------------------------------------- | ------------- |
| 1     | Load, validate, clean, USD conversion, split                    | `src/data.py`, `src/utils.py`                 | Done          |
| 2     | Exploratory data analysis                                       | `src/eda.py`                                  | Done          |
| 3     | Baseline LightGBM + MLflow                                      | `src/baseline.py`, `src/metrics.py`           | **In review** |
| 4     | Text model (DistilBERT)                                         | `src/text_model.py`                           | –             |
| 5     | Image embeddings (CLIP)                                         | `src/image.py`                                | –             |
| 6     | Fusion network (concat vs gated)                                | `src/fusion.py`                               | –             |
| 7     | Ablation + significance tests                                   | `src/evaluate.py`                             | –             |
| 8     | SHAP, Integrated Gradients, conformal intervals, review routing | `src/explain.py`                              | –             |
| 9     | FastAPI + web page + monitoring + Docker                        | `app/main.py`, `app/index.html`, `Dockerfile` | –             |

## Results

| Model               | MAE | MdAPE | Within 10% |
| ------------------- | --- | ----- | ---------- |
| Baseline (LightGBM) | TBD | TBD   | TBD        |
| Best fusion model   | TBD | TBD   | TBD        |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m src.data audit
python -m src.data build          # all markets, prices in USD
python -m src.eda
python -m src.baseline
mlflow ui --backend-store-uri sqlite:///mlflow.db   # view runs
```

## Project files

```
config.yaml              all settings
src/utils.py             config, seed, logging, MLflow
src/metrics.py           MAE, MdAPE, within-10% (all models)
src/data.py              Phase 1 pipeline
src/eda.py               Phase 2 EDA
src/baseline.py          Phase 3 LightGBM baseline
tests/test_data.py       Phase 1 tests
tests/test_eda.py        Phase 2 tests
tests/test_baseline.py   Phase 3 tests
docs/PHASE_1.md          Phase 1 guide
docs/PHASE_2.md          Phase 2 guide
docs/PHASE_3.md          Phase 3 guide
docs/GIT.md              how to commit and push each phase
.github/workflows/ci.yml lint + tests on every push
```

Data: [`MongoDB/airbnb_embeddings`](https://huggingface.co/datasets/MongoDB/airbnb_embeddings) (Apache-2.0). Data files are git-ignored and rebuilt by the pipeline.
