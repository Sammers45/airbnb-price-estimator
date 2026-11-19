# Phase 10: Fixing the form-only skew, README, and shipping to `main`

## 1. The fix: ranges sized for what the web form really sends

**Problem (found by Phase 9 monitoring).** The web form asks for about ten fields. Everything
else (review scores, host details, coordinates) takes the market's typical value. Replaying
300 listings that way showed:

|                    | Full listing data | Form only |
| ------------------ | ----------------- | --------- |
| MdAPE              | 18.2%             | 28.5%     |
| 80% range coverage | 87.3%             | **71.7%** |
| Sent to review     | 12.0%             | **4.3%**  |

The Phase 8 ranges were sized on errors made with full data, so they were too narrow for
form-only predictions. Worse, the difficulty model saw "typical" inputs, judged the listings
easy, and stopped sending them to review: less accurate, yet flagged less.

**Fix (`src/form_ranges.py`).** Measure the errors the service makes on form-only input,
honestly, and size the ranges on those. For each fold k:

1. train LightGBM on the other folds with full data, as in production;
2. rebuild every fold-k listing exactly as the web form sends it. This uses the same
   `serve.listing_row` function as the API, with typical values taken from the other folds only;
3. predict → out-of-fold form-only predictions.

Phase 8's calibrator (difficulty model + conformal range sizes + review threshold) is then
fit on these predictions and saved to `models/intervals_form/`. The service uses it for every
web request; replays with full data keep the Phase 8 ranges. Each response says which it used
(`range_basis`), and so does the page.

The report compares **before** (Phase 8 ranges applied to form-only predictions, as Phase 9
did) with **after**, all out-of-fold.

## 2. Files

| File                        | Change                                                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/form_ranges.py`        | **new**: the fix above                                                                                                                                              |
| `tests/test_form_ranges.py` | **new**: 3 tests (form rebuild, honest coverage, service uses the new ranges)                                                                                       |
| `src/serve.py`              | `listing_row` and `listing_from_row` are module functions (reused by Phase 10); `build_form` accepts a DataFrame; the form-only calibrator is used for web requests |
| `src/monitor.py`            | `simulate` uses `listing_from_row`                                                                                                                                  |
| `web/index.html`            | shows which ranges were used                                                                                                                                        |
| `README.md`                 | rewritten: results, architecture, decisions, how to run                                                                                                             |

## 3. Run

```powershell
pytest                                            # 111 tests
python -m src.form_ranges                         # about 1 minute
python -m src.serve build
python -m src.monitor simulate --n 300 --fresh --form-only
```

After the fix, the form-only replay should show coverage near 80% and a review rate near 20%.
Its accuracy is still about 28% MdAPE, because a form can't replace a listing's history. That
is honest, and the wider ranges and higher review rate now say so.

## 4. Shipping to `main`

Each phase branch was created from the one before, so `phase-10` contains all ten phases.

```powershell
git checkout phase-9
git checkout -b phase-10
git add src/form_ranges.py tests/test_form_ranges.py src/serve.py src/monitor.py web/index.html README.md docs/PHASE_10.md reports/
git commit -m "Phase 10: ranges sized for form-only input; README"
git push -u origin phase-10
```

Then merge into `main` with a pull request. On GitHub, open the repo, click **Compare & pull
request** for `phase-10`, base `main`, and wait for the green CI check. Then click **Merge**.
A merged PR with passing CI is good to show on a portfolio. From the command line instead:

```powershell
git checkout main
git pull
git merge --no-ff phase-10 -m "Merge phases 1-10"
git push origin main
git tag -a v1.0 -m "Airbnb price estimator v1.0"
git push origin v1.0
```

Optional cleanup, once `main` has everything:
`git branch -d phase-1 phase-2 ...` and `git push origin --delete phase-1 ...`.

## 5. PySpark (decided: not used)

The data is 5,555 rows; Polars cleans it in under a second. Spark would add a JVM, Hadoop
`winutils` on Windows, and a cluster abstraction with nothing to distribute. The README's
"Scaling" note says where Spark _would_ fit: Phase 1 cleaning and batch scoring at millions of
listings, reusing the same logic. Saying "I measured it and chose not to" is a stronger
interview answer than a Spark job that runs on one laptop.
