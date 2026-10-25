"""Phase 2: exploratory data analysis of the cleaned listings (Polars).

Usage:
    python -m src.eda

Reads data/processed/listings.parquet (from Phase 1) and writes
reports/eda.md, reports/eda_summary.json and reports/figures/eda_*.png.

Prices differ a lot between markets (Istanbul vs Maui), so most analyses use
the *within-market* relative price:

    rel_log_price = log_price - median(log_price) of the listing's market

A feature that only tells you "this is Maui" then gets no credit; what is left
is what the feature says about price inside a market.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from src.data import md  # noqa: E402
from src.utils import load_config, set_seed, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- helpers
def add_relative_price(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``rel_log_price``: log price minus its market's median log price."""
    return df.with_columns(
        (pl.col("log_price") - pl.col("log_price").median().over("market")).alias("rel_log_price")
    )


def add_distance_to_centre(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``dist_centre_km``: great-circle distance to the market's median location.

    A candidate feature for Phase 3: central listings usually cost more.
    """
    lat0 = pl.col("latitude").median().over("market").radians()
    lon0 = pl.col("longitude").median().over("market").radians()
    lat, lon = pl.col("latitude").radians(), pl.col("longitude").radians()
    a = ((lat - lat0) / 2).sin() ** 2 + lat.cos() * lat0.cos() * ((lon - lon0) / 2).sin() ** 2
    return df.with_columns((2 * EARTH_RADIUS_KM * a.sqrt().arcsin()).alias("dist_centre_km"))


def numeric_feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric model inputs (excludes the target, ids, folds, flags and amenity dummies)."""
    skip = {"price", "log_price", "rel_log_price", "price_local", "fold", "latitude", "longitude"}
    return [
        c
        for c, t in df.schema.items()
        if t.is_numeric() and c not in skip and not c.startswith(("amen_", "has_"))
    ]


# --------------------------------------------------------------------------- analyses
def target_summary(df: pl.DataFrame) -> dict[str, Any]:
    """Distribution of the USD price and why we model log(price)."""
    p = df["price"]
    return {
        "n": df.height,
        "median_usd": round(float(p.median()), 2),
        "mean_usd": round(float(p.mean()), 2),
        "p01_usd": round(float(p.quantile(0.01)), 2),
        "p99_usd": round(float(p.quantile(0.99)), 2),
        "max_usd": round(float(p.max()), 2),
        "skew_price": round(float(p.skew()), 2),
        "skew_log_price": round(float(df["log_price"].skew()), 2),
    }


def suspicious_prices(df: pl.DataFrame, z: float) -> pl.DataFrame:
    """Listings whose price is extreme for their market (robust z-score of log price).

    robust_z = (log_price - market median) / (1.4826 * market MAD). Nothing is
    dropped here: Phase 3 applies the fold-wise outlier rule, and these rows are
    reviewed by hand.
    """
    mad = (pl.col("log_price") - pl.col("log_price").median().over("market")).abs()
    scale = 1.4826 * mad.median().over("market")
    robust_z = pl.when(scale > 0).then(pl.col("rel_log_price") / scale).otherwise(0.0)
    cols = [
        "listing_id",
        "market",
        "room_type",
        "property_type",
        "accommodates",
        "bedrooms",
        "currency",
        "price_local",
        "price",
    ]
    return (
        df.with_columns(robust_z.round(2).alias("robust_z"))
        .filter(pl.col("robust_z").abs() > z)
        .select([c for c in cols if c in df.columns] + ["robust_z"])
        .with_columns(pl.col("price").round(2))
        .sort("robust_z", descending=True)
    )


def price_by(df: pl.DataFrame, col: str, top: int | None = None) -> pl.DataFrame:
    """Listing count and USD price quartiles per category, largest groups first."""
    out = (
        df.group_by(col)
        .agg(
            pl.len().alias("rows"),
            pl.col("price").median().round(1).alias("median_usd"),
            pl.col("price").quantile(0.25).round(1).alias("q25_usd"),
            pl.col("price").quantile(0.75).round(1).alias("q75_usd"),
        )
        .sort("rows", descending=True)
    )
    return out.head(top) if top else out


def spearman_with_price(df: pl.DataFrame, cols: list[str], target: str) -> pl.DataFrame:
    """Spearman correlation of each column with ``target`` (pairwise complete rows)."""
    rows = []
    for c in cols:
        pair = df.select(pl.col(c).cast(pl.Float64), pl.col(target)).drop_nulls()
        if pair.height < 10 or pair[c].n_unique() < 2:
            continue
        rho = pair.select(pl.corr(c, target, method="spearman")).item()
        rows.append(
            {
                "feature": c,
                "spearman": round(float(rho), 3),
                "n": pair.height,
                "missing_pct": round(100 * (1 - pair.height / df.height), 1),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "spearman": pl.Float64,
                "n": pl.Int64,
                "missing_pct": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort(pl.col("spearman").abs(), descending=True)


def amenity_lift(df: pl.DataFrame) -> pl.DataFrame:
    """For each amenity: within-market price difference between listings with and without it.

    lift_pct = exp(median rel price with - median rel price without) - 1.
    """
    rows = []
    for c in (c for c in df.columns if c.startswith("amen_")):
        has = df.filter(pl.col(c) == 1)["rel_log_price"]
        hasnt = df.filter(pl.col(c) == 0)["rel_log_price"]
        if has.len() < 10 or hasnt.len() < 10:
            continue
        diff = float(has.median()) - float(hasnt.median())
        rows.append(
            {
                "amenity": c.removeprefix("amen_"),
                "share_pct": round(100 * has.len() / df.height, 1),
                "lift_pct": round(100 * (np.exp(diff) - 1), 1),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={"amenity": pl.String, "share_pct": pl.Float64, "lift_pct": pl.Float64}
        )
    return pl.DataFrame(rows).sort(pl.col("lift_pct").abs(), descending=True)


def missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Null share per column, only columns that have nulls."""
    nulls = df.null_count().transpose(
        include_header=True, header_name="column", column_names=["missing"]
    )
    return (
        nulls.filter(pl.col("missing") > 0)
        .with_columns((100 * pl.col("missing") / df.height).round(1).alias("pct"))
        .sort("missing", descending=True)
    )


def image_coverage(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Share of listings with an image embedding per market, and whether missing photos
    relate to price (if they do, a missing photo is itself informative)."""
    by_market = (
        df.group_by("market")
        .agg(
            pl.len().alias("rows"),
            (100 * pl.col("has_image_emb").mean()).round(1).alias("with_image_pct"),
        )
        .sort("rows", descending=True)
    )
    med = df.group_by("has_image_emb").agg(pl.col("rel_log_price").median())
    lookup = dict(zip(med["has_image_emb"].to_list(), med["rel_log_price"].to_list(), strict=True))
    gap = None
    if 0 in lookup and 1 in lookup:
        gap = round(100 * (np.exp(lookup[0] - lookup[1]) - 1), 1)
    return by_market, {"missing_photo_price_gap_pct": gap}


def text_length(df: pl.DataFrame) -> dict[str, float]:
    """How much description text there is to learn from."""
    n = df["text_n_words"]
    return {
        "median_words": float(n.median()),
        "p90_words": float(n.quantile(0.9)),
        "under_10_words_pct": round(100 * float((n < 10).mean()), 1),
        "over_256_words_pct": round(100 * float((n > 256).mean()), 1),  # DistilBERT ~ 512 tokens
    }


def embedding_matrix(df: pl.DataFrame, col: str) -> np.ndarray:
    """List column of equal-length vectors -> 2-D float32 array, each row scaled to unit length."""
    width = int(df[col].list.len().max())
    x = df[col].list.to_array(width).to_numpy().astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def linear_probe(x: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas: list[float]) -> float:
    """Out-of-fold R² of a ridge regression on ``x``, using the Phase 1 folds.

    A quick check of how much price signal an embedding holds before any deep model.
    The ridge strength is chosen inside each training fold (RidgeCV), so the
    validation fold never influences it.
    """
    pred = np.empty_like(y, dtype=float)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        model = RidgeCV(alphas=alphas).fit(x[tr], y[tr])
        pred[va] = model.predict(x[va])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return round(1 - ss_res / ss_tot, 3)


def run_probes(df: pl.DataFrame, alphas: list[float]) -> pl.DataFrame:
    """R² on log price and on within-market price for each modality."""
    rows = []
    y_abs, y_rel = df["log_price"].to_numpy(), df["rel_log_price"].to_numpy()
    folds = df["fold"].to_numpy()
    market = df.select(pl.col("market").fill_null("?")).to_dummies().to_numpy().astype(np.float32)
    rows.append(
        {
            "input": "market only",
            "rows": df.height,
            "r2_log_price": linear_probe(market, y_abs, folds, alphas),
            "r2_within_market": 0.0,
        }
    )
    if "text_embeddings" in df.columns:
        x = embedding_matrix(df, "text_embeddings")
        rows.append(
            {
                "input": "text embedding",
                "rows": df.height,
                "r2_log_price": linear_probe(x, y_abs, folds, alphas),
                "r2_within_market": linear_probe(x, y_rel, folds, alphas),
            }
        )
    if "image_embeddings" in df.columns and df["has_image_emb"].sum() >= 50:
        sub = df.filter(pl.col("has_image_emb") == 1)
        x = embedding_matrix(sub, "image_embeddings")
        f = sub["fold"].to_numpy()
        rows.append(
            {
                "input": "image embedding",
                "rows": sub.height,
                "r2_log_price": linear_probe(x, sub["log_price"].to_numpy(), f, alphas),
                "r2_within_market": linear_probe(x, sub["rel_log_price"].to_numpy(), f, alphas),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return f"figures/{path.name}"


def plot_market_boxes(df: pl.DataFrame, path: Path, min_rows: int = 30) -> str:
    """Box plot of USD price (log axis) per market, ordered by median."""
    big = price_by(df, "market").filter(pl.col("rows") >= min_rows).sort("median_usd")
    names = big["market"].to_list()
    data = [df.filter(pl.col("market") == m)["price"].to_numpy() for m in names]
    fig, ax = plt.subplots(figsize=(0.8 * len(names) + 2, 5))
    ax.boxplot(data, showfliers=False)  # vertical boxes: works on every Matplotlib version
    ax.set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set(ylabel="nightly price, USD (log scale)", title="Price by market (outliers hidden)")
    return _save(fig, path)


def plot_bars(table: pl.DataFrame, label: str, value: str, title: str, path: Path) -> str:
    """Horizontal bar chart, largest absolute value on top."""
    t = table.head(15).reverse()
    fig, ax = plt.subplots(figsize=(8, 0.35 * t.height + 1.5))
    vals = t[value].to_numpy()
    ax.barh(t[label].to_list(), vals, color=np.where(vals >= 0, "#3b6ea8", "#c0504d"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=title, xlabel=value)
    return _save(fig, path)


def plot_price_vs_guests(df: pl.DataFrame, path: Path) -> str:
    """Median within-market price by number of guests, one line per room type."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    g = (
        df.filter(pl.col("accommodates") <= 12)
        .group_by("room_type", "accommodates")
        .agg(pl.col("rel_log_price").median(), pl.len().alias("n"))
        .filter(pl.col("n") >= 10)
        .sort("accommodates")
    )
    for (room,), part in g.group_by("room_type", maintain_order=True):
        ax.plot(part["accommodates"], np.exp(part["rel_log_price"]), marker="o", label=room)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set(
        xlabel="guests (accommodates)",
        ylabel="price vs market median (x)",
        title="Within-market price by number of guests and room type",
    )
    ax.legend()
    return _save(fig, path)


# --------------------------------------------------------------------------- report
def write_report(path: Path, s: dict[str, Any], t: dict[str, pl.DataFrame], figs: dict) -> None:
    ts = s["target"]
    lines = [
        "# Exploratory data analysis",
        "_Generated by `python -m src.eda`. Do not edit by hand._",
        "\nAll prices are USD (converted in Phase 1). *Within-market* means relative to the "
        "listing's market median, so market-level differences don't count as feature signal.",
        "\n## 1. Target",
        f"- {ts['n']} listings, median **${ts['median_usd']}**, mean ${ts['mean_usd']}, "
        f"1st-99th percentile ${ts['p01_usd']}-${ts['p99_usd']}, max ${ts['max_usd']}.",
        f"- Skewness: price {ts['skew_price']}, log price {ts['skew_log_price']}. "
        "Models predict `log_price` and convert back.",
        "\n## 2. Prices that are extreme for their market",
        f"Robust z-score of log price within market above {s['suspect_z']}: "
        f"**{t['suspicious'].height}** listings ({s['suspicious_pct']}%). Not dropped; "
        "the fold-wise outlier rule handles training, and these are checked by hand.",
        md(t["suspicious"].head(s["top_n"])) if t["suspicious"].height else "_None flagged._",
        "\n## 3. Price by market",
        md(t["by_market"]),
        f"\n![price by market]({figs['markets']})",
        "\n## 4. Price by room type and property type",
        md(t["by_room"]),
        "\n",
        md(t["by_property"]),
        f"\n![price vs guests]({figs['guests']})",
        "\n## 5. Numeric features vs within-market price (Spearman)",
        md(t["spearman"]),
        f"\n![correlations]({figs['spearman']})",
        "\n## 6. Amenities: within-market price lift",
        "Lift is a correlation, not a causal effect (pools tend to come with bigger homes).",
        md(t["amenities"].head(15)),
        f"\n![amenity lift]({figs['amenities']})",
        "\n## 7. Missing values (processed table)",
        md(t["missing"]),
        "\n## 8. Images and text",
        "- Listings without a photo embedding are priced "
        f"{s['image']['missing_photo_price_gap_pct']}% differently (within market) "
        "from those with one.",
        md(t["image_by_market"]),
        f"\n- Description length: median {s['text']['median_words']} words, 90th percentile "
        f"{s['text']['p90_words']}; {s['text']['under_10_words_pct']}% have under 10 words, "
        f"{s['text']['over_256_words_pct']}% have over 256 (may be truncated by the Transformer).",
        "\n## 9. Signal check: linear probes (out-of-fold R²)",
        "Ridge regression on each input alone, using the Phase 1 folds. A rough lower bound on "
        "what each modality can add; the deep models in Phases 4-6 should do better.",
        md(t["probes"]),
        "\n## 10. Candidate feature: distance to market centre",
        f"Spearman with within-market price: **{s['dist_centre_spearman'] or 'n/a'}** "
        "(negative = central listings cost more).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every analysis, save figures, the markdown report and a JSON summary."""
    set_seed(cfg["seed"])
    e = cfg["eda"]
    reports = Path(cfg["paths"]["reports"])
    figdir = reports / "figures"
    df = pl.read_parquet(cfg["paths"]["processed"])
    df = add_distance_to_centre(add_relative_price(df))
    log.info("EDA on %d rows x %d columns", df.height, df.width)

    num_cols = numeric_feature_columns(df)
    tables = {
        "suspicious": suspicious_prices(df, e["suspect_z"]),
        "by_market": price_by(df, "market"),
        "by_room": price_by(df, "room_type"),
        "by_property": price_by(df, "property_type", top=10),
        "spearman": spearman_with_price(df, num_cols, "rel_log_price"),
        "amenities": amenity_lift(df),
        "missing": missing_values(df.drop([c for c in df.columns if c.endswith("_embeddings")])),
        "probes": run_probes(df, e["probe_alphas"]),
    }
    tables["image_by_market"], image = image_coverage(df)
    dist = tables["spearman"].filter(pl.col("feature") == "dist_centre_km")["spearman"]
    summary = {
        "target": target_summary(df),
        "suspect_z": e["suspect_z"],
        "top_n": e["top_n"],
        "suspicious_pct": round(100 * tables["suspicious"].height / df.height, 2),
        "image": image,
        "text": text_length(df),
        "dist_centre_spearman": dist[0] if dist.len() else None,
    }
    figs = {
        "markets": plot_market_boxes(df, figdir / "eda_price_by_market.png"),
        "guests": plot_price_vs_guests(df, figdir / "eda_price_vs_guests.png"),
        "spearman": plot_bars(
            tables["spearman"],
            "feature",
            "spearman",
            "Spearman with within-market log price",
            figdir / "eda_spearman.png",
        ),
        "amenities": plot_bars(
            tables["amenities"],
            "amenity",
            "lift_pct",
            "Within-market price lift (%)",
            figdir / "eda_amenities.png",
        ),
    }
    with pl.Config(tbl_width_chars=1000):  # don't wrap column names in markdown tables
        write_report(reports / "eda.md", summary, tables, figs)
    summary["tables"] = {k: v.head(30).to_dicts() for k, v in tables.items()}
    (reports / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"EDA report: {reports / 'eda.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 EDA")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    run(load_config(args.config))


if __name__ == "__main__":
    main()