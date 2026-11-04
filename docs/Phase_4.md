# Phase 4: Text model (fine-tuned multilingual DistilBERT) on Google Colab

`src/text_model.py` fine-tunes a Transformer on each listing's text (name + summary + description, with prices masked in Phase 1) to predict `log1p(price)`. Training needs a GPU, so it runs on a **free Google Colab T4** with `notebooks/phase4_text_colab.ipynb`. Everything else runs on your computer.

## Design decisions

| Decision                                                                         | Why                                                                                                                                  |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `distilbert-base-multilingual-cased` instead of English DistilBERT               | 8 of the 12 markets often write in Turkish, Spanish, Portuguese, French or Chinese. An English-only model would read those as noise. |
| Mean pooling over tokens + linear head                                           | Uses every token rather than just the first. Gives a stable 768-number text vector per listing.                                      |
| Same 5 folds as the baseline, best epoch picked on 10% of each **training** fold | Fair comparison, with no validation-fold leakage                                                                                     |
| Target standardised with training-fold mean and std                              | Stable training. Validation statistics are never used.                                                                               |
| Same fold-wise outlier rule as the baseline                                      | Like-for-like comparison                                                                                                             |
| Saves out-of-fold **predictions and text vectors**                               | Each listing's vector comes from a model that never saw it, so Phase 6 can fuse it with the table safely.                            |
| Final model on all rows, saved in fp16                                           | For the web app. Half the download size.                                                                                             |
| Text only: no structured features                                                | Measures what the text alone knows. Combining comes in Phase 6.                                                                      |

Don't expect the text model to beat the baseline on its own. Its value is the information the table doesn't have, which Phase 6 adds on top and Phase 7 tests for significance.

## Steps

### On your computer

```powershell
git checkout main
git pull
git checkout -b phase-4
# copy in the Phase 4 files
pip install -r requirements.txt       # adds torch (CPU on Windows) and transformers
pytest                                # 53 tests; the 8 new ones use a tiny model, no download
git add src tests docs notebooks config.yaml requirements.txt .github README.md
git commit -m "feat: phase 4 text model (multilingual DistilBERT) + Colab notebook"
git push -u origin phase-4            # Colab downloads the code from this branch
```

### On Google Colab

1. Go to https://colab.research.google.com, then **File → Upload notebook** and pick `notebooks/phase4_text_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order. You'll upload `data\processed\listings.parquet` in step 4.
4. Training takes about 30–40 minutes. Download `phase4_results.zip` at the end.

### Back on your computer

```powershell
# extract phase4_results.zip inside S:\AirBNB\airbnb-price-estimator
python -m src.text_model log          # records the run in local MLflow
git add reports
git commit -m "docs: phase 4 text model results"
git push
```

## Outputs

| Path                                                                        | Git? | Used by                                        |
| --------------------------------------------------------------------------- | ---- | ---------------------------------------------- |
| `reports/text_model.md`, `reports/text_model_metrics.json`                  | yes  | review, comparison with baseline               |
| `data/predictions/text_oof.parquet` (prediction + 768-d vector per listing) | no   | fusion (Phase 6), significance tests (Phase 7) |
| `models/text_model/` (about 270 MB)                                         | no   | web app (Phase 9)                              |

## Settings (`config.yaml → text_model`)

`max_length: 256`, `batch_size: 32`, `epochs: 3` (maximum), `learning_rate: 3.0e-5`, `warmup_frac: 0.1`, `weight_decay: 0.01`. If Colab runs out of GPU memory, set `batch_size: 16`.

## Files in this phase

| File                                                                                                    | Status                                             |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `src/text_model.py`, `tests/test_text_model.py`, `notebooks/phase4_text_colab.ipynb`, `docs/PHASE_4.md` | new                                                |
| `config.yaml`                                                                                           | changed: `text_model:` section at the end          |
| `requirements.txt`                                                                                      | changed: `torch`, `transformers`                   |
| `.github/workflows/ci.yml`                                                                              | changed: installs the small CPU-only PyTorch first |
| `README.md`                                                                                             | changed: status table and baseline results         |
