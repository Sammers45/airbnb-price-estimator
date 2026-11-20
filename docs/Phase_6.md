# Phase 6: Fusion network (table + text + photo), runs on your CPU

`src/fusion.py` combines everything built so far and compares two architectures. This is the "architectural improvement" part of the project.

## Inputs (all out-of-fold, joined on `listing_id`)

| Input                                           | From                                                            | Size                       |
| ----------------------------------------------- | --------------------------------------------------------------- | -------------------------- |
| Table features (numbers, amenities, categories) | `data/processed/listings.parquet` (Phase 1)                     | ~60 numbers + 6 categories |
| LightGBM prediction                             | `data/predictions/baseline_oof.parquet` (Phase 3)               | 1                          |
| Text vector                                     | `data/predictions/text_oof.parquet` (Phase 4, Colab)            | 768                        |
| Photo vector + prompt scores                    | processed file + `data/predictions/image_oof.parquet` (Phase 5) | 512 + 12                   |

If a file is missing, the script tells you which phase to run.

## Two architectures

```
            table ──► block ─┐                     table ─► token ─┐
             text ──► block ─┼─► join ─► MLP         text ─► token ─┼─► self-attention ─► attention pooling ─► head
            photo ──► block ─┘                      photo ─► token ─┘   (tokens interact)    (learns how much to
                                                    (masked if none)                          trust each input)
        A. concatenation                         B. attention fusion
```

- **A. Concatenation**: each input becomes 64 numbers, they're joined, and a small MLP predicts. A missing photo is zeroed, and a has-photo flag is added.
- **B. Attention fusion**: each input becomes a 64-number token with a learned "which input am I" embedding. One self-attention layer lets the inputs inform each other (for example, the text can change how the photo is read). Then a learned query attends over the tokens and outputs **weights per listing**, saved and summarised in the report. A missing photo is **masked**, so its weight is exactly 0.

## Design choices

| Choice                                                                                                                     | Why                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Residual on the baseline**: output = LightGBM prediction + learned correction, with the correction starting at exactly 0 | Neural networks usually lose to gradient boosting on table data. This makes the question "do text and photos _improve_ the baseline?", and the network starts as good as the baseline. |
| **Modality dropout (10%)**: text or photo randomly hidden during training                                                  | Robust to missing photos and weak descriptions, including in the web app                                                                                                               |
| Same 5 folds, fold-wise outlier rule, early stopping on 10% of the **training** fold                                       | Same rules as every other model                                                                                                                                                        |
| **3 seeds averaged per fold**                                                                                              | Neural nets vary run to run; averaging gives stable numbers for Phase 7's significance tests                                                                                           |
| Table preprocessing fit per training fold                                                                                  | Medians, scaling and category ids never see validation rows                                                                                                                            |

**Stacking caveat (worth knowing for interviews):** the text vectors and baseline predictions are out-of-fold, but each came from a model trained on the _other_ folds, which include the current fusion validation fold. This is standard stacking practice and can make fusion scores very slightly optimistic. It's noted in the report.

## Run

```powershell
git checkout main
git pull
git checkout -b phase-6
# save src\fusion.py, tests\test_fusion.py, docs\PHASE_6.md; append the config section
pytest                         # 72 tests (62 + 10 new)
python -m src.fusion train     # both architectures, about 7-15 minutes on CPU
```

For Phase 7 ablations, the same script can use fewer inputs, for example `python -m src.fusion train --modalities tab,text`.

## Outputs

| Path                                                                         | Git? | Used by                                       |
| ---------------------------------------------------------------------------- | ---- | --------------------------------------------- |
| `reports/fusion.md`, `reports/fusion_metrics.json`                           | yes  | review, resume numbers                        |
| `data/predictions/fusion_concat_oof.parquet`, `fusion_attention_oof.parquet` | no   | Phase 7 significance tests, Phase 8 intervals |
| `models/fusion_concat/`, `models/fusion_attention/`                          | no   | web app (Phase 9)                             |

## Commit

```powershell
git add src/fusion.py tests/test_fusion.py docs/PHASE_6.md config.yaml reports
git commit -m "feat: phase 6 fusion network (concatenation vs attention, residual on baseline)"
git push -u origin phase-6
```

## Send back for approval

`reports/fusion.md`. Section 1 is the headline: baseline vs text-only vs the two fusion architectures, all on the same 5,555 listings.
