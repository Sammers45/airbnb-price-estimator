# Phase 5: Image features (CLIP ViT-B/32), runs on your CPU

`src/image_model.py` turns each listing's main photo into model inputs. No GPU or Colab is needed.

## Key fact

The dataset already contains a 512-number vector per photo. Its README says these were made with **`openai/clip-vit-base-patch32`** (CLIP ViT-B/32) from each listing's `picture_url`. So we use those vectors directly, and we **verify** that we can reproduce them. The web app (Phase 9) has to embed uploaded photos the same way, or its predictions would be meaningless.

## Steps

| Step   | Command                            | What it does                                                                                                                                                                                                                                                  |
| ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Verify | `python -m src.image_model verify` | Downloads the CLIP model once (about 600 MB), re-embeds about 40 real listing photos, and compares them with the dataset's vectors using cosine similarity (1.0 = identical). Expected result: **MATCH** (median ≥ 0.95). Also embeds the text prompts below. |
| Train  | `python -m src.image_model train`  | Image-only price model plus prompt scores on the Phase 1 folds                                                                                                                                                                                                |

## What "train" produces

1. **Image-only model**: ridge regression on the 512-d photo vector (unit length), with the strength chosen inside each training fold. It's compared with the Phase 3 baseline **on the same listings**. Don't expect it to beat the baseline alone; its job is to add to it in Phase 6.
2. **Readable photo features (zero-shot CLIP prompts)**: for each photo, how much it looks like phrases such as _"a luxurious, high-end home interior"_, _"a room with a sea view"_ or _"a small, cramped room"_, relative to the other phrases. The report shows how each one correlates with **within-market** price. In Phase 8, SHAP can then say things like _"the photo looks luxurious: +$18"_. You can edit or add prompts in `config.yaml → image_model.prompts`, then rerun `verify` and `train`.
3. **`data/predictions/image_oof.parquet`**: one row per listing with `has_image`, the out-of-fold image prediction and the `img_*` prompt scores. They're empty for the 17% of listings without a photo vector, and fusion handles those with a mask.

## Leakage notes

- Ridge strength is picked inside each training fold, and validation rows are never seen.
- Prompt scores use no prices at all (only the photo and fixed phrases), so they can't leak the target. Their correlations in the report are descriptive.

## Run

```powershell
git checkout main
git pull
git checkout -b phase-5
# save src\image_model.py, tests\test_image_model.py, docs\PHASE_5.md; append the config section
pytest                                   # 62 tests (53 + 9 new, no downloads)
python -m src.image_model verify         # needs internet; about 2 minutes on CPU
python -m src.image_model train          # about 1 minute
```

## Outputs

| Path                                                                                      | Git? |
| ----------------------------------------------------------------------------------------- | ---- |
| `reports/image_model.md`, `reports/image_model_metrics.json`, `reports/image_verify.json` | yes  |
| `data/predictions/image_oof.parquet`                                                      | no   |
| `models/clip_prompts.json`                                                                | no   |

## Commit

```powershell
git add src/image_model.py tests/test_image_model.py docs/PHASE_5.md config.yaml reports
git commit -m "feat: phase 5 CLIP image features (verified vectors, zero-shot prompt scores)"
git push -u origin phase-5
```

## Send back for approval

`reports/image_model.md`. Check above all that section 1 says **match**. If it says no match, stop and tell me before Phase 6.
