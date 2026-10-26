"""Phase 3: LightGBM baseline on structured features, tracked in MLFlow.

Usage:
    python -m src.baseline
    
Every later model must beat this one. The script:
    1. runs 5-fold cross-validation on the Phase 1 folds, predicting log(price);
    2. compares it with a naive "market median" predictor;
    3. saves one out-of-fold predictions per listing (used for paired tests in Phase 7);
    4. trains a final model on all rows for the web app (phase 9);
    5. logs params, metrics and artifacts to MLFlow and writes reports/baseline.md.
    
Leakage rules: category codes, market centres and the outlier rule are fit on the training
fold only; early stopping uses a slice of the training fold, never the validation fold.
Text and image inputs are excluded.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data import md, outlier_bounds
from src.metrics import metrics_by, price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)

CATEGORICAL = [
    "market",
    "room_type",
    "property_type",
    "bed_type",
    "cancellation_policy",
    "neighbourhood"
]

EXCLUDED = {
    "listing_id",
    "country",
    "currency",
    "price_local",
    "price",
    "log_price",
    "fold",
    "text",
    "text_n_words",
    "picture_url",
    "text_embeddings",
    "image_embeddings",
    "has_text_emb",
    "has_image_emb"
}
EARTH_RADIUS_KM = 6371.0

# ------------------------------------------------------------- Features
def haversine_km(lat: pl.Expr, lon: pl.Expr, lat0: pl.Expr, lon0: pl.Expr) -> pl.Expr:
    """Great-circle distance in km between two points given in danger"""
    la ,lo, la0, lo0 = lat.radians(), lon.radians(), lat0.radians(), lon0.radians()
    a = ((la - la0) / 2).sin() ** 2 + la.cos() * la0.cos() * ((lo - lo0) / 2).sin() ** 2
    return 2 * EARTH_RADIUS_KM * a.sqrt().arcsin()

@dataclass
class FeatureBuilder:
    """Turns a listings frame into a numeric matrix. Fit on training rows only.
    
    Learned State:
        categroies: the category values seen in training, per categorical column
        centres:    median latitude/longitude per market, from training rows
        numeric:    numeric input columns
    """

    categories: dict[str, list[str]] = field(default_factory=dict)
    centres: dict[str, list[float]] = field(default_factory=dict)
    numeric: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + ["has_reviews", "dist_centre_km"] + list(self.categories)

    @property
    def categorical_names(self) -> list[str]:
        return list(self.categories)

    def fit(self, df: pl.DataFrame) -> FeatureBuilder:
        self.categories = {
            c: sorted(df[c].drop_nulls().cast(pl.String).unique().to_list())
            for c in CATEGORICAL
            if c in df.columns
        }
        centres = df.group_by("market").agg(
            pl.col("latitude").median(), pl.col("longitude").median()
        )
        self.centres = {
            r["market"]: [r["latitude"], r["longitude"]]
            for r in centres.iter_rows(named=True)
            if r["market"] is not None
        }
        self.numeric = [
            c
            for c, t in df.schema.items()
            if t.is_numeric() and c not in EXCLUDED and c not in CATEGORICAL
        ]
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Feature frame in ``feature_name`` order. Unseen categories become null"""
        lat0 = pl.col("market").replace_strict(
            {m: c[0] for m, c in self.centres.items()}, default=None, return_dtype=pl.Float64
        )
        lon0 = pl.col("market").replace_strict(
            {m: c[1] for m, c in self.centres.items()}, default=None, return_dtype=pl.Float64
        )
        exprs = [pl.col(c).cast(pl.Float64) for c in self.numeric]
        exprs.append(
            pl.col("days_since_first_review").is_not_null().cast(pl.Float64).alias("has_reviews")
        )
        exprs.append(
            haversine_km(pl.col("latitude"), pl.col("longitude"), lat0, lon0).alias(
                "dist_centre_km"
            )
        )
        for c,value in self.categories.items():
            codes = {v: float(i) for i, v in enumerate(value)}
            exprs.append(
                pl.col(c)
                .cast(pl.String)
                .replace_strict(codes, default=None, return_dtype=pl.Float64)
                .alias(c)
            )
        return df.select(exprs)

    def matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Float matrix (nulls -> NaN, whcih LightGBM treats as missing)"""
        return self.transform(df).to_numpy().astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureBuilder:
        return cls(**d)


## ---------------------------------------------------------------------- TRAINING
def lgb_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """LightGBM params from config, plus fixed settings for reproducibility"""
    return {
        **cfg["baseline"]["params"],
        "seed": cfg["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }

def fit_booster(
        x: np.ndarray, y: np.ndarray, fb: FeatureBuilder, cfg: dict[str, Any], seed: int
) -> lgb.Booster:
    """Train with early stopping on a random slice of the TRAINING rows."""
    b = cfg["baseline"]
    rng = np.random.default_rng(seed)
    inner = rng.random(len(y)) < b["inner_valid_frac"]
    names, cats = fb.feature_names, fb.categorical_names
    train = lgb.Dataset(
        x[~inner], y[~inner], feature_name=names, categorical_feature=cats, free_raw_data=False
    )
    valid = lgb.Dataset(x[inner], y[inner], reference=train)
    return lgb.train(
        lgb_params(cfg),
        train,
        num_boost_round=b["num_boost_round"],
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(b["early_stopping_rounds"], verbose=False)],
    )

def drop_outliers(train: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Fold-wise outlier rule from Phase 1, applied to training rows only."""
    o = cfg["outliers"]
    lo, hi = outlier_bounds(train["log_price"].to_numpy(), o["lower_quantile"], o["upper_quantile"])
    return train.filter(pl.col("log_price").is_between(lo, hi))

def naive_market_median(train: pl.DataFrame, valid: pl.DataFrame) -> np.ndarray:
    """Predict each lsiting's training-market median price (global median if unseen)."""
    med = train.group_by("market").agg(pl.col("price").median().alias("naive"))
    out = valid.select("market").join(med, on="market", how="left")["naive"]
    return out.fill_null(float(train["price"].median())).to_numpy()

def cross_validate(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Out-of-fold predictions ofr every listing, plus per-fold details."""
    folds = sorted(df["fold"].unique().to_list())
    parts, fold_rows, importances, best_iters = [], [], [], []
    for k in folds:
        train = drop_outliers(df.filter(pl.col("fold") != k), cfg)
        valid = df.filter(pl.col("fold") == k)
        fb = FeatureBuilder().fit(train)
        booster = fit_booster(
            fb.matrix(train), train["log_price"].to_numpy(), fb, cfg, cfg["seed"] + k        
        )
        pred = np.expm1(booster.predict(fb.matrix(valid), num_iteration=booster.best_iteration))
        naive = naive_market_median(train, valid)
        parts.append(
            valid.select("listing_id", "fold", "market", "room_type", "price").with_columns(
                pl.Series("pred", pred), pl.Series("naive_pred", naive)
            )
        )
        m = price_metrics(valid["price"].to_numpy(), pred)
        fold_rows.append(
            {
                "fold": k,
                "train_rows": train.height,
                "valid_rows": valid.height,
                "best_iteration": booster.best_iteration,
                **m,
            }
        )
        importances.append(
            dict(zip(fb.feature_names, booster.feature_importance("gain").tolist(), strict=True))
        )
        best_iters.append(booster.best_iteration)
        log.info("fold %d: MAE $%.2f, within 10%% %.1f%%", k, m["mae_usd"], m["within_10_pct"])
    oof = pl.concat(parts).sort("listing_id")
    imp = (
        pl.DataFrame(importances)
        .mean()
        .transpose(include_header=True, header_name="feature", column_names=["gain"])
    )
    imp = imp.with_columns((100 * pl.col("gain") / pl.col("gain").sum()).round(2).alias("gain_pct"))
    details = {
        "folds": pl.DataFrame(fold_rows),
        "importance": imp.sort("gain", descending=True),
        "best_iterations": best_iters
    }
    return oof, details

def train_final(
        df: pl.DataFrame, cfg: dict[str, Any], n_rounds: int
) -> tuple[lgb.Booster, FeatureBuilder]:
    """MOdel on all rows for serving; number of trees = average best iteration from CV."""
    train = drop_outliers(df, cfg)
    fb = FeatureBuilder().fit(train)
    data = lgb.Dataset(
        fb.matrix(train),
        train["log_price"].to_numpy(),
        feature_name = fb.feature_names,
        categorical_feature = fb.categorical_names,
    )
    return lgb.train(lgb_params(cfg), data, num_boost_round=n_rounds), fb


## ------------------------------------------------------------------------------- REPORT
def plot_pred_vs_actual(oof: pl.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(oof["price"], oof["pred"], s=4, alpha=0.3)
    lims = [max(1.0, float(oof["price"].min())), float(oof["price"].max())]
    ax.plot(lims, lims, color='black', lw=0.8)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="actual price (USD)",
        ylabel="predicted (USD)",
        title="LightGBM baseline: out-of-fold predictions",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"figures/{path.name}"

def write_report(path: Path, s: dict[str, Any], t: dict[str, pl.DataFrame], fig: str) -> None:
    lines = [
        "# Baseline: LightGBM on structured features",
        "_Generated by `python -m src.baseline`. Do not efit by hand._",
        f"\n{s['rows']} listings, {s['n_folds']}-fold CV on the Phase 1 folds, "
        f"target `log1p(price)`, {s['n_features']} features. "
        "Metrics are on out-of-fold predictions in USD.",
        "\n## 1. Overall",
        md(t["overall"]),
        f"\nMAE across folds: ${s['fold_mae_mean']} ± {s['fold_mae_std']} (mean ± std).",
        "\n## 2. Per fold",
        md(t["folds"]),
        "\n## 3. Per Market",
        md(t["by_market"]),
        "\n## 4. Per room type",
        md(t["by_room"]),
        "\n## 5. Feature importance (gain, mean over folds, top 20)",
        md(t["importance"].head(20)),
        f"\n![predicted vs actual]({fig})",
        f"\n## 6. Final model\nTrained on all rows with {s['final_rounds']} trees "
        f"(mean best iteration). Saved to `{s['model_path']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cross-validate, train the final model, save everything and log it to MLFlow."""  
    set_seed(cfg["seed"])
    paths = cfg["paths"]
    reports, preds_dir, models_dir = (
        Path(paths["reports"]),
        Path(paths["predictions"]),
        Path(paths["models"]),
    )
    for p in (reports / "figures", preds_dir, models_dir):
        p.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(paths["processed"])
    log.info("Baseline on %d rows", df.height)

    with mlflow_run(cfg, "baseline_lightgbm") as mlflow:
        oof, details = cross_validate(df, cfg)
        y, pred, naive = (
            oof["price"].to_numpy(),
            oof["pred"].to_numpy(),
            oof["naive_pred"].to_numpy(),
        ) 
        overall = pl.DataFrame(
            [
                {"model": "naive: market median", **price_metrics(y, naive)},
                {"model": "LightGBM baseline", **price_metrics(y, pred)},
            ]
        )
        final_rounds = max(1, int(np.mean(details['best_iterations'])))
        booster, fb = train_final(df, cfg, final_rounds)

        oof_path = preds_dir / "baseline_oof.parquet"
        model_path = models_dir / "baseline_lightgbm.txt"
        feat_path = models_dir / "baseline_features.json"
        oof.write_parquet(oof_path)
        booster.save_model(str(model_path))
        feat_path.write_text(json.dumps(fb.to_dict(), indent=2), encoding="utf-8")

        fold_mae = details["folds"]["mae_usd"]
        summary = {
            "rows": df.height,
            "n_folds": len(details["best_iterations"]),
            "n_features": len(fb.feature_names),
            "overall": overall.to_dicts(),
            "fold_mae_mean": round(float(fold_mae.mean()), 2),
            "fold_mae_std": round(float(fold_mae.std()), 2),
            "final_rounds": final_rounds,
            "model_path": str(model_path).replace("\\", "/"),
        }
        tables = {
            "overall": overall,
            "folds": details["folds"],
            "by_market": metrics_by(oof, "market"),
            "by_room": metrics_by(oof, "room_type"),
            "importance": details["importance"].select("feature", "gain_pct"), 
        }
        fig = plot_pred_vs_actual(oof, reports / "figures" / "baseline_pred_vs_actual.png")
        with pl.Config(tbl_width_chars=1000):
            write_report(reports / "baseline.md", summary, tables, fig)
        (reports / "baseline_metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        if mlflow is not None:
            mlflow.log_params({f"lgb_{k}": v for k, v in cfg["baseline"]["params"].items()})
            mlflow.log_params(
                {
                    "n_features": len(fb.feature_names),
                    "final_rounds": final_rounds,
                    "rows": df.height,
                    "outlier_q": str(list(cfg["outliers"].values())),
                }
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in price_metrics(y, pred).items()})
            mlflow.log_metrics({f"naive_{k}": v for k, v in price_metrics(y, naive).items()})
            for row in details["folds"].iter_rows(named=True):
                mlflow.log_metric("fold_mae_usd", row["mae_usd"], step=row["fold"])
            for p in  (
                reports / "baseline.md",
                reports / "baseline_metrics.json",
                reports / "figures" / fig.split("/")[-1],
                model_path,
                feat_path,
                oof_path,
            ):
                mlflow.log_artifact(str(p))

    best = overall.row(1, named=True)
    print(
        f"LightGBM baseline: MAE ${best['mae_usd']}, MdAPE {best["mdape_pct"]}%, "
        f"within 10% {best['within_10_pct']}%. Report: {reports / 'baseline.md'}"
    )
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 LightGBm baseline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    run(load_config(args.config))

if __name__ == "__main__":
    main()