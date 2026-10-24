"""Phase 1: load, validate, clean and split the Airbnb data (Polars).

Usage:
    python -m src.data audit                      # checks + market recommendation
    python -m src.data build --market "<name>"    # clean, split, validate, save

The two steps are separate because prices are in local currency per market,
and choosing one market or pooling all of them is your decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.utils import load_config, set_seed, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

MONEY_COLUMNS = ["price", "security_deposit", "cleaning_fee", "extra_people",
                 "weekly_price", "monthly_price"]  # fmt: skip
NUMERIC_COLUMNS = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
                   "maximum_nights", "guests_included", "number_of_reviews"]  # fmt: skip
CATEGORICAL_COLUMNS = ["property_type", "room_type", "bed_type", "cancellation_policy"]

# Explicit money amounts in text are replaced so the text model can't read the answer.
PRICE_TOKEN = "[PRICE]"
_SYM = r"(?:R\$|HK\$|A\$|C\$|US\$|\$|€|£|¥|₺)"
_NUM = r"\d[\d,.]*"
_WORDS = r"(?:usd|eur|euros?|gbp|pounds?|dollars?|bucks|reais|brl|hkd|aud|cad|lira|tl|try|yuan|rmb)"
PRICE_PATTERNS = [
    rf"(?i){_SYM}\s?{_NUM}",  # $95, € 200
    rf"(?i)\b{_NUM}\s?{_SYM}",  # 80€
    rf"(?i)\b{_NUM}\s?{_WORDS}\b",  # 80 euros, 300 TL
    rf"(?i)\b{_NUM}\s?(?:/|per)\s?(?:night|week|month)\b",  # 50/night
]


# --------------------------------------------------------------------------- loading
def download(dataset_id: str) -> pl.DataFrame:
    """Read the dataset from Hugging Face.

    First choice: Polars reads Hugging Face's auto-converted Parquet copy directly.
    Fallback: the `datasets` library parses the original 245 MB JSON file.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless warning on Windows
    url = f"hf://datasets/{dataset_id}@~parquet/**/*.parquet"
    try:
        log.info("Reading %s with Polars", url)
        return pl.read_parquet(url)
    except Exception as err:  # network, auth or path problems -> try the other route
        log.warning("Polars read failed (%s); falling back to `datasets`", err)
        from datasets import load_dataset  # imported here so tests need no network

        return load_dataset(dataset_id, split="train").to_polars()


def load_raw(cfg: dict[str, Any]) -> pl.DataFrame:
    """Download the dataset once, cache it as Parquet, then always read the cache."""
    path = Path(cfg["paths"]["raw"])
    if not path.exists():
        df = download(cfg["data"]["hf_dataset"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    df = pl.read_parquet(path)
    log.info("Raw data: %d rows x %d columns", df.height, df.width)
    return df


# --------------------------------------------------------------------------- parsing
def nested(df: pl.DataFrame, *path: str) -> tuple[pl.Expr, pl.DataType]:
    """Expression for a (possibly nested) field, or a null literal if it doesn't exist."""
    dtype = df.schema.get(path[0])
    if dtype is None:
        return pl.lit(None), pl.Null()
    expr = pl.col(path[0])
    for key in path[1:]:
        fields = {f.name: f.dtype for f in dtype.fields} if isinstance(dtype, pl.Struct) else {}
        if key not in fields:
            return pl.lit(None), pl.Null()
        expr, dtype = expr.struct.field(key), fields[key]
    return expr, dtype


def to_float(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Convert numbers, decimals, strings like "$1,200.00" or {"$numberDecimal": ..} to Float64."""
    if isinstance(dtype, pl.Struct):  # MongoDB extended JSON
        first = dtype.fields[0]
        return to_float(expr.struct.field(first.name), first.dtype)
    if dtype == pl.String:
        return expr.str.replace_all(",", "").str.extract(r"(-?\d+(?:\.\d+)?)", 1).cast(pl.Float64)
    return expr.cast(pl.Float64, strict=False)


def to_flag(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Convert booleans or "t"/"f" strings to 1.0 / 0.0 (null stays null)."""
    if dtype == pl.String:
        lowered = expr.str.to_lowercase()
        return (
            pl.when(lowered.is_in(["t", "true", "yes", "1"]))
            .then(1.0)
            .when(lowered.is_in(["f", "false", "no", "0"]))
            .then(0.0)
        )
    return expr.cast(pl.Float64, strict=False)


def to_datetime(df: pl.DataFrame, col: str) -> pl.Expr:
    """Datetime expression for a string/date/datetime column (null if missing)."""
    dtype = df.schema.get(col)
    if dtype == pl.String:
        return pl.col(col).str.to_datetime(strict=False).dt.replace_time_zone(None)
    if dtype == pl.Date:
        return pl.col(col).cast(pl.Datetime("us"))
    if isinstance(dtype, pl.Datetime):
        return pl.col(col).dt.replace_time_zone(None).cast(pl.Datetime("us"))
    return pl.lit(None, dtype=pl.Datetime("us"))


# --------------------------------------------------------------------------- features
def flatten(df: pl.DataFrame) -> pl.DataFrame:
    """Pull location, host, review-score and photo fields out of the nested columns."""
    field = lambda *p: nested(df, *p)[0]  # noqa: E731
    coords, cdt = nested(df, "address", "location", "coordinates")
    if isinstance(cdt, pl.List | pl.Array):
        coords = coords.cast(pl.List(pl.Float64))
        lat, lon = coords.list.get(1, null_on_oob=True), coords.list.get(0, null_on_oob=True)
    else:
        lat = lon = pl.lit(None, dtype=pl.Float64)

    exprs = [
        field("address", "country").alias("country"),
        field("address", "market").alias("market"),
        pl.coalesce(field("address", "suburb"), field("address", "government_area"))
        .cast(pl.String)
        .alias("neighbourhood"),
        lat.alias("latitude"),
        lon.alias("longitude"),
        field("images", "picture_url").cast(pl.String).alias("picture_url"),
        to_float(*nested(df, "host", "host_listings_count")).alias("host_listings_count"),
        to_flag(*nested(df, "host", "host_is_superhost")).alias("host_is_superhost"),
        to_flag(*nested(df, "host", "host_identity_verified")).alias("host_identity_verified"),
    ]
    scores = df.schema.get("review_scores")
    if isinstance(scores, pl.Struct):
        for f in scores.fields:
            name = "rs_" + f.name.replace("review_scores_", "")
            exprs.append(
                to_float(pl.col("review_scores").struct.field(f.name), f.dtype).alias(name)
            )
    return df.with_columns(exprs)


def parse_numbers(df: pl.DataFrame) -> pl.DataFrame:
    """Price and numeric columns -> Float64, whatever type they arrived as."""
    cols = [c for c in ["price", *NUMERIC_COLUMNS] if c in df.columns]
    return df.with_columns(to_float(pl.col(c), df.schema[c]).alias(c) for c in cols)


def date_features(df: pl.DataFrame) -> pl.DataFrame:
    """Days since first/last review, measured from the scrape date (no future information)."""
    scraped = to_datetime(df, "last_scraped")
    return df.with_columns(
        (scraped - to_datetime(df, "first_review")).dt.total_days().cast(pl.Float64)
        .alias("days_since_first_review"),
        (scraped - to_datetime(df, "last_review")).dt.total_days().cast(pl.Float64)
        .alias("days_since_last_review"),
    )  # fmt: skip


def amenity_vocabulary(df: pl.DataFrame, top_k: int) -> list[str]:
    """Most frequent amenities. Chosen by frequency only (never price), so no target leakage."""
    if "amenities" not in df.columns:
        return []
    counts = (
        df.select(pl.col("amenities").list.unique().explode().drop_nulls().alias("a"))
        .group_by("a")
        .len()
        .sort(["len", "a"], descending=[True, False])
    )
    return counts["a"].head(top_k).to_list()


def amenity_features(df: pl.DataFrame, vocab: list[str]) -> pl.DataFrame:
    """``amenities_count`` plus one 0/1 ``amen_<name>`` column per vocabulary item."""
    if "amenities" not in df.columns:
        return df.with_columns(pl.lit(0).alias("amenities_count"))
    exprs = [pl.col("amenities").list.len().fill_null(0).alias("amenities_count")]
    used: set[str] = set()
    for name in vocab:
        col = "amen_" + (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "x")
        while col in used:  # "Wi-Fi" and "Wi Fi" slugify the same
            col += "_x"
        used.add(col)
        exprs.append(
            pl.col("amenities").list.contains(name).fill_null(False).cast(pl.Int8).alias(col)
        )
    return df.with_columns(exprs)


def strip_prices(text: pl.Expr) -> pl.Expr:
    """Replace explicit money amounts with ``[PRICE]`` and squash whitespace."""
    for pattern in PRICE_PATTERNS:
        text = text.str.replace_all(pattern, PRICE_TOKEN)
    return text.str.replace_all(r"\s+", " ").str.strip_chars()


def combine_text(df: pl.DataFrame, fields: list[str]) -> pl.Expr:
    """Join the text fields with " . " (skipping empty ones), then strip prices."""
    parts = []
    for f in fields:
        if f in df.columns:
            c = pl.col(f).cast(pl.String).str.strip_chars()
            parts.append(pl.when(c != "").then(c))
    if not parts:
        return pl.lit("")
    return strip_prices(pl.concat_str(parts, separator=" . ", ignore_nulls=True).fill_null(""))


def embedding_ok(df: pl.DataFrame, col: str, dim: int) -> pl.Expr:
    """1 if the row has a finite embedding of the right size, else 0."""
    if col not in df.columns:
        return pl.lit(0, dtype=pl.Int8)
    vec = pl.col(col).cast(pl.List(pl.Float32))
    good = (vec.list.len() == dim) & vec.list.eval(pl.element().is_finite()).list.all()
    return good.fill_null(False).cast(pl.Int8)


def to_usd(df: pl.DataFrame, cfg: dict[str, Any], col: str = "price") -> pl.DataFrame:
    """Add ``currency`` and ``price_local``, and convert ``col`` to USD.

    Raises ValueError if a country has no currency in the config, so a new
    market can never be silently mixed in at the wrong scale.
    """
    cur = cfg["currency"]
    unknown = set(df["country"].drop_nulls().unique()) - set(cur["by_country"])
    if unknown or df["country"].null_count():
        raise ValueError(f"No currency configured for countries: {sorted(unknown) or ['<null>']}")
    rates = {c: cur["units_per_usd"][code] for c, code in cur["by_country"].items()}
    return df.with_columns(
        pl.col("country").replace_strict(cur["by_country"]).alias("currency"),
        pl.col(col).alias("price_local"),
        (pl.col(col) / pl.col("country").replace_strict(rates, return_dtype=pl.Float64)).alias(col),
    )


def clean(raw: pl.DataFrame, cfg: dict[str, Any], market: str | None) -> tuple[pl.DataFrame, dict]:
    """Raw nested records -> flat, leakage-free table. Returns (frame, info for the report)."""
    if market is None:
        raise ValueError('No market chosen. Run the audit, then use --market "<name>" or "all".')
    steps = [("raw", raw.height)]

    df = parse_numbers(flatten(raw))
    if market != "all":
        df = df.filter(pl.col("market") == market)
        if df.height == 0:
            raise ValueError(f"No rows for market {market!r}; check the audit table spelling.")
    steps.append((f"market = {market}", df.height))

    df = to_usd(df, cfg)  # every later step (and every model) works in USD

    df = df.filter(pl.col("price").is_not_null() & (pl.col("price") >= cfg["data"]["min_price"]))
    steps.append(("drop missing / non-positive price", df.height))

    df = df.with_columns(pl.col("_id").cast(pl.String).alias("listing_id")).unique(
        "listing_id", keep="first", maintain_order=True
    )
    steps.append(("drop duplicate listings", df.height))

    vocab = amenity_vocabulary(df, cfg["features"]["top_k_amenities"])
    df = amenity_features(date_features(df), vocab)
    text = combine_text(df, cfg["features"]["text_fields"])
    chk = cfg["checks"]
    df = df.with_columns(
        text.alias("text"),
        embedding_ok(df, "text_embeddings", chk["text_emb_dim"]).alias("has_text_emb"),
        embedding_ok(df, "image_embeddings", chk["image_emb_dim"]).alias("has_image_emb"),
        pl.col("price").log1p().alias("log_price"),
    ).with_columns(pl.col("text").str.split(" ").list.len().alias("text_n_words"))

    keep = (
        ["listing_id", "country", "market", "neighbourhood", "latitude", "longitude"]
        + [c for c in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS if c in df.columns]
        + ["host_listings_count", "host_is_superhost", "host_identity_verified"]
        + sorted(c for c in df.columns if c.startswith("rs_"))
        + ["days_since_first_review", "days_since_last_review"]
        + [c for c in df.columns if c == "amenities_count" or c.startswith("amen_")]
        + ["text", "text_n_words", "picture_url", "has_text_emb", "has_image_emb"]
        + [c for c in ("text_embeddings", "image_embeddings") if c in df.columns]
        + ["currency", "price_local", "price", "log_price"]
    )
    leakage = set(cfg["features"]["leakage_columns"])
    df = df.select([c for c in keep if c not in leakage])
    for c in ("text_embeddings", "image_embeddings"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.List(pl.Float32)))
    info = {"steps": steps, "amenities": vocab,
            "leakage_removed": sorted(leakage & set(raw.columns))}  # fmt: skip
    return df, info


# --------------------------------------------------------------------------- splits
def make_folds(
    y: np.ndarray, n_folds: int, n_bins: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Fold id per row, stratified on price quantile bins (and market when pooling markets)."""
    y = np.asarray(y)
    n_bins = max(1, min(n_bins, len(y) // n_folds))
    ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    strata = (ranks * n_bins // len(y)).astype(str)
    if groups is not None:  # combine market + bin, unless that stratum is too small
        combo = np.char.add(np.char.add(np.asarray(groups, dtype=str), "|"), strata)
        _, inverse, counts = np.unique(combo, return_inverse=True, return_counts=True)
        strata = np.where(counts[inverse] >= n_folds, combo, strata)
    # Any stratum still smaller than n_folds is merged into one "rare" group; if even that
    # is too small, it joins the largest stratum. Every stratum then fills all folds.
    labels, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    strata = np.where(counts[inverse] >= n_folds, strata, "rare")
    if 0 < (strata == "rare").sum() < n_folds:
        strata = np.where(strata == "rare", labels[counts.argmax()], strata)
    folds = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(skf.split(np.zeros(len(y)), strata)):
        folds[idx] = k
    return folds


def outlier_bounds(y_train_log: np.ndarray, lower_q: float, upper_q: float) -> tuple[float, float]:
    """Outlier rule, fit on a TRAINING fold only. Training rows outside are dropped (Phase 3+);
    validation rows are never dropped, so metrics always cover every listing."""
    return float(np.quantile(y_train_log, lower_q)), float(np.quantile(y_train_log, upper_q))


# --------------------------------------------------------------------------- checks
def _url_works(url: str, timeout: float) -> bool:
    import requests

    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            return r.status_code < 300
    except requests.RequestException:
        return False


def check_image_urls(
    urls: list[str | None], n: int, timeout: float, threshold: float, seed: int,
    fetch: Callable[[str, float], bool] = _url_works,
) -> dict[str, Any]:  # fmt: skip
    """Test n random photo URLs (missing URL = failure)."""
    sample = random.Random(seed).sample(urls, min(n, len(urls)))
    with ThreadPoolExecutor(max_workers=10) as ex:
        ok = list(ex.map(lambda u: bool(u) and fetch(u, timeout), sample))
    failed = ok.count(False)
    rate = failed / max(len(sample), 1)
    return {"tested": len(sample), "failed": failed, "fail_rate": round(rate, 3),
            "use_provided_image_embeddings": rate > threshold}  # fmt: skip


def validate(df: pl.DataFrame, cfg: dict[str, Any]) -> None:
    """Data contract for the processed table. Raises ValueError listing every violation."""
    errors = []
    if df.height == 0:
        errors.append("table is empty")
    if df["listing_id"].n_unique() != df.height:
        errors.append("listing_id is not unique")
    price = df["price"]
    if price.null_count() or (price <= 0).any() or not price.is_finite().all():
        errors.append("price must be finite and > 0")
    elif not np.allclose(df["log_price"].to_numpy(), np.log1p(price.to_numpy())):
        errors.append("log_price != log1p(price)")
    leaked = set(cfg["features"]["leakage_columns"]) & set(df.columns)
    if leaked:
        errors.append(f"leakage columns present: {sorted(leaked)}")
    if "fold" in df.columns and set(df["fold"].unique()) != set(range(cfg["cv"]["n_folds"])):
        errors.append("fold ids are incomplete")
    if errors:
        raise ValueError("Data contract failed: " + "; ".join(errors))


# --------------------------------------------------------------------------- reporting
def md(df: pl.DataFrame) -> str:
    """Polars DataFrame -> markdown table."""
    with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, tbl_rows=-1, tbl_cols=-1,
                   fmt_str_lengths=120):  # fmt: skip
        return str(df)


def price_stats(prices: pl.Series) -> pl.DataFrame:
    return prices.drop_nulls().describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


def save_histogram(prices: pl.Series, title: str, path: Path) -> None:
    """Histogram of price (up to p99) and of log1p(price)."""
    p = prices.drop_nulls().filter(prices.drop_nulls() > 0).to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if p.size:
        ax[0].hist(p[p <= np.quantile(p, 0.99)], bins=50)
        ax[1].hist(np.log1p(p), bins=50)
    ax[0].set(title=f"{title}: price (<= p99)", xlabel="price")
    ax[1].set(title=f"{title}: log1p(price)", xlabel="log1p(price)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(cfg: dict[str, Any], audit: dict[str, Any], build: dict | None) -> Path:
    """Write reports/data_audit.md from the audit (and build, once it has run)."""
    lines = [
        "# Data audit",
        "_Generated by `python -m src.data`. Do not edit by hand._",
        f"\nRaw rows: **{audit['n_rows']}**, columns: {audit['n_cols']}, seed: {cfg['seed']}",
        "\n## 1. Data types of money fields",
        md(pl.DataFrame(audit["money_types"])),
        "\n## 2. Rows per country and market (median in local currency and in USD)",
        md(pl.DataFrame(audit["markets"])),
        f"\nRecommended market (largest, if >= {cfg['data']['min_market_rows']} rows, "
        f"else all): **{audit['recommended_market']}**",
        "\n## 3. Image URL check",
        "Failed {failed} of {tested} ({fail_rate:.0%}). Use provided image embeddings: "
        "**{use_provided_image_embeddings}**".format(**audit["url_check"]),
        "\n## 4. Raw price distribution (all markets, mixed currencies)",
        md(pl.DataFrame(audit["price_stats"])),
        "\n![raw prices](figures/price_raw.png)",
        "\n## 5. Missing values (raw, top 25)",
        md(pl.DataFrame(audit["missing"]).head(25)),
    ]
    if build is None:
        lines.append("\n## 6. Processed data\nTBD - run the `build` step.")
    else:
        lines += [
            f"\n## 6. Chosen market: **{build['market']}**",
            "\nAll prices are converted to **USD** with 2019 average exchange rates "
            "(see `currency` in config.yaml). `price_local` keeps the original value.",
            "\n## 7. Cleaning steps",
            md(pl.DataFrame(build["steps"], schema=["step", "rows"], orient="row")),
            f"\nLeakage columns removed: {', '.join(build['leakage_removed'])}.",
            "\nPrice mentions in the text were replaced with `[PRICE]`. The provided "
            "`text_embeddings` come from the raw `space` field and can't be scrubbed, so "
            "treat them as a possible weak leakage source.",
            "\n## 8. Processed price distribution",
            md(pl.DataFrame(build["price_stats"])),
            "\n![processed prices](figures/price_processed.png)",
            "\n## 9. Outlier rule",
            f"Inside each training fold, keep log1p(price) between the fold's "
            f"{cfg['outliers']['lower_quantile']} and {cfg['outliers']['upper_quantile']} "
            "quantiles. Validation rows are never dropped. Nothing is dropped in Phase 1.",
            "\n## 10. Folds (stratified on price deciles)",
            md(pl.DataFrame(build["folds"])),
            f"\n## 11. Embeddings present\ntext: {build['has_text_emb']} / {build['n_rows']}, "
            f"image: {build['has_image_emb']} / {build['n_rows']}",
            f"\n## 12. Top amenities\n{', '.join(build['amenities'])}",
        ]
    path = Path(cfg["paths"]["reports"]) / "data_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- steps
def run_audit(cfg: dict[str, Any], raw: pl.DataFrame, url_check=check_image_urls) -> dict:
    """Step 1: the four checks on raw data, then stop so you can choose the market."""
    set_seed(cfg["seed"])
    flat = flatten(raw)
    prices = to_float(pl.col("price"), raw.schema["price"]).alias("price")
    priced = flat.with_columns(prices.alias("price_num"))
    if set(priced["country"].drop_nulls().unique()) <= set(cfg["currency"]["by_country"]):
        priced = to_usd(priced, cfg, col="price_num").rename({"price_num": "price_usd"})
    else:  # an unknown country: show local prices only, the build step will stop on it
        priced = priced.with_columns(pl.lit(None, dtype=pl.Float64).alias("price_usd"))
        priced = priced.rename({"price_num": "price_local"})
    markets = (
        priced.group_by("country", "market")
        .agg(
            pl.len().alias("rows"),
            pl.col("price_local").median().round(2).alias("median_local"),
            pl.col("price_usd").median().round(2).alias("median_usd"),
        )
        .sort("rows", descending=True)
    )
    top = markets.row(0, named=True) if markets.height else {"rows": 0}
    recommended = top["market"] if top["rows"] >= cfg["data"]["min_market_rows"] else "all"
    chk = cfg["checks"]
    url = url_check(
        flat["picture_url"].to_list(), chk["url_sample"], chk["url_timeout_s"],
        chk["url_fail_threshold"], cfg["seed"],
    )  # fmt: skip
    raw_prices = raw.select(prices)["price"]
    reports = Path(cfg["paths"]["reports"])
    save_histogram(raw_prices, "all markets", reports / "figures" / "price_raw.png")

    missing = (
        raw.null_count()
        .transpose(include_header=True, header_name="column", column_names=["missing"])
        .with_columns((pl.col("missing") / raw.height * 100).round(1).alias("pct"))
        .sort("missing", descending=True)
    )
    audit = {
        "n_rows": raw.height,
        "n_cols": raw.width,
        "money_types": [
            {"column": c, "dtype": str(raw.schema.get(c, "missing"))} for c in MONEY_COLUMNS
        ],  # fmt: skip
        "markets": markets.to_dicts(),
        "recommended_market": recommended,
        "url_check": url,
        "price_stats": price_stats(raw_prices).to_dicts(),
        "missing": missing.to_dicts(),
    }
    (reports / "phase1_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    write_report(cfg, audit, None)
    print(md(markets))
    print(f"\nImage URLs failed: {url['failed']}/{url['tested']}")
    print(f"Recommended market: {recommended}")
    print('Next: python -m src.data build --market "<market or all>"')
    return audit


def run_build(cfg: dict[str, Any], raw: pl.DataFrame, market: str | None) -> pl.DataFrame:
    """Step 2: clean, add folds, validate, save the table and finish the report."""
    set_seed(cfg["seed"])
    audit_file = Path(cfg["paths"]["reports"]) / "phase1_audit.json"
    if not audit_file.exists():
        raise FileNotFoundError("Run `python -m src.data audit` first.")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    df, info = clean(raw, cfg, market)
    groups = df["market"].to_numpy() if market == "all" else None
    folds = make_folds(df["log_price"].to_numpy(), cfg["cv"]["n_folds"],
                       cfg["cv"]["n_price_bins"], cfg["seed"], groups)  # fmt: skip
    df = df.with_columns(pl.Series("fold", folds))
    validate(df, cfg)

    out = Path(cfg["paths"]["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    save_histogram(df["price"], f"market={market}",
                   Path(cfg["paths"]["reports"]) / "figures" / "price_processed.png")  # fmt: skip
    fold_table = (
        df.group_by("fold")
        .agg(pl.len().alias("rows"), pl.col("price").median().alias("median_price"))
        .sort("fold")
    )
    build = {
        **info,
        "market": market,
        "n_rows": df.height,
        "price_stats": price_stats(df["price"]).to_dicts(),
        "folds": fold_table.to_dicts(),
        "has_text_emb": int(df["has_text_emb"].sum()),
        "has_image_emb": int(df["has_image_emb"].sum()),
    }
    report = write_report(cfg, audit, build)
    print(f"Saved {df.height} rows to {out}. Report: {report}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 data pipeline")
    parser.add_argument("step", choices=["audit", "build"])
    parser.add_argument("--market", help='Market name from the audit, or "all"')
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    raw = load_raw(cfg)
    if args.step == "audit":
        run_audit(cfg, raw)
    else:
        run_build(cfg, raw, args.market or cfg["data"]["market"])


if __name__ == "__main__":
    main()