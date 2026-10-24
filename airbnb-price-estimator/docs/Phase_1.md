# Phase 1: Load, validate, clean, split

Everything is in `src/data.py` and uses **Polars**.

## Steps

| Step         | Function     | What it does                                                                                                                                                                                                                                              |
| ------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Load         | `load_raw`   | Reads [`MongoDB/airbnb_embeddings`](https://huggingface.co/datasets/MongoDB/airbnb_embeddings) (5,555 rows) once with Polars from Hugging Face's Parquet copy, falls back to the `datasets` library if needed, and caches it as `data/raw/airbnb.parquet` |
| Audit        | `run_audit`  | Money-field types, rows per market, a 50-URL photo check, price stats, missing values                                                                                                                                                                     |
| _You decide_ |              | Pick one market, or `all` (prices are in local currency)                                                                                                                                                                                                  |
| Clean        | `clean`      | Flattens nested fields, parses prices to numbers, filters the market, drops bad prices, adds features, removes leakage                                                                                                                                    |
| Split        | `make_folds` | 5 folds stratified on price deciles (and market when using `all`)                                                                                                                                                                                         |
| Validate     | `validate`   | Fails if ids repeat, a price is invalid, a leakage column remains or a fold is missing                                                                                                                                                                    |
| Save         | `run_build`  | `data/processed/listings.parquet` and `reports/data_audit.md`                                                                                                                                                                                             |

## Features created

- **Location:** `latitude`, `longitude`, `neighbourhood`
- **Host:** `host_is_superhost`, `host_identity_verified`, `host_listings_count`
- **Reviews:** `rs_*` sub-scores, `days_since_first_review`, `days_since_last_review` (measured from the scrape date)
- **Amenities:** `amenities_count` plus 30 `amen_*` flags
- **Text:** `text` (name + summary + description, with prices replaced by `[PRICE]`) and `text_n_words`
- **Embeddings:** `text_embeddings` and `image_embeddings` are kept, plus `has_text_emb` and `has_image_emb` flags
- **Target:** `log_price = log1p(price)`

## Leakage controls

- Removed: `weekly_price`, `monthly_price`, `cleaning_fee`, `security_deposit`, `extra_people`, `availability`, `reviews`, `listing_url` and `calendar_last_scraped`. `_id` is kept as `listing_id`.
- Prices written in the text are masked.
- The provided `text_embeddings` were built from the raw `space` field, which can't be scrubbed. The ablation in Phase 7 will check them.
- The outlier rule (`outlier_bounds`) is fit inside each training fold only, starting in Phase 3.

## Run

```bash
pip install -r requirements.txt
pytest                                           # offline, synthetic data
python -m src.data audit                         # prints the market table
python -m src.data build --market "<name>"       # or --market all
```

## Send back for approval

1. The market table printed by `audit`.
2. The line showing how many image URLs failed.
3. `reports/data_audit.md` once `build` has run.
