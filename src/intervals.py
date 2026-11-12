"""Phase 8a: price ranges (conformal predictions) and human-review routing. Runs on a CPU.

Usage:
    python -m src.intervals

For every listing we give a price RANGE that should contain the true price 80% (and 90%)
of the time, built from the Phase 7 best model's out-of-fold predictions:

- constant: the same +/- band on the log scale for every listing.
- adaptive: a small LightGBM ("difficulty model") predicts how wrong the price model is
likely to be for each listing; the band is scaled by that, so ordinary listings get
narrow ranges and unusual ones get wide ranges.

Honest evaluation: for each fold, the difficulty model and the band size are fit on the
OTHER folds only; coverage is then measured on the held-out fold.

Routing: a listing goes to a person when its 80% range is unusually wide or its market
has too few listings to rent.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.baseline import FeatureBuilder, lgb_params
from src.data import md
from src.metrics import price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)
EPS = 0.01 # 1% error floor, so log(|error|) is defined for near-perfect predictions
OWN_COLUMNS = ["pred", "pred_log", "resid"]

## ------------------------------------------------------------------------- INPUTS
def load_inputs(cfg: dict[str, Any]) -> pl.DataFrame:
    """Processed listings + the pHase 7 best model's out-of-fold predictions"""
    path = Path(cfg["paths"]["predictions"]) / "best_oof.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run Phase 7 first")
    best = pl.read_parquet(path).select("listing_id", "pred")
    df = pl.read_parquet(cfg["paths"]["processed"]).join(best, on="listing_id", how="inner")
    return df.with_columns(
        pl.col("market").fill_null(""),
        pl.col("pred").log1p().alias("pred_log"),
    ).with_columns((pl.col("log_price") - pl.col("pred_log")).alias("resid"))

def  conformal_q(scores: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction ceiL((n + 1) * level) / n."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * level)
    return float(s[min(k, len(s)) - 1])

def band(pred_log: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-scale centre +/- half width -> (low, high) prices in USD."""
    return np.clip(np.expm1(pred_log - half), 0, None), np.expm1(pred_log + half)


## ------------------------------------------------------------------------------------ CALIBRATOR
@dataclass
class Calibrator:
    """Everything needed to put a range around a new prediction."""

    fb: FeatureBuilder
    booster: lgb.Booster
    levels: list[float]
    q_constant: dict[str, float]
    q_adaptive: dict[str, float]
    width_threshold: float
    market_counts: dict[str, int]
    min_market_rows: int

    def features(self, df: pl.DataFrame) -> np.ndarray:
        return np.hstack([self.fb.matrix(df), df["pred_log"].to_numpy()[:, None]])

    def difficulty(self, df: pl.DataFrame) -> np.ndarray:
        """Expected absolute log error of the price model for each listing."""
        return np.exp(self.booster.predict(self.features(df)))

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ranges for every level and method, plus the routing decision."""
        p, sig = df["pred_log"].to_numpy(), self.difficulty(df)
        cols: dict[str, np.ndarray] = {"difficulty": sig}
        for lv in self.levels:
            k = f"{round(lv * 100)}"
            cols[f"lo_const_{k}"], cols[f"hi_const_{k}"] = band(
                p, np.full_like(p, self.q_constant[k])
            )
            cols[f"lo_{k}"], cols[f"hi_{k}"] = band(p, self.q_adaptive[k] * sig)
        k = f"{round(self.levels[0] * 100)}"
        width = (cols[f"hi_{k}"] - cols[f"lo_{k}"]) / np.expm1(p)
        rare = np.array(
            [self.market_counts.get(m, 0) < self.min_market_rows for m in df["market"].to_list()]
        )
        wide = width > self.width_threshold
        reason = np.where(rare, "rare market", np.where(wide, "wide range", ""))
        return df.select("listing_id").with_columns(
            *[pl.Series(c, v) for c, v in cols.items()],
            pl.Series("rel_width", width),
            pl.Series("review", rare | wide),
            pl.Series("reason", reason),
        )

    def save(self, out: Path, source: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out / "difficulty_lightgbm.txt"))
        meta = {
            "source_model": source,
            "levels": self.levels,
            "q_constant": self.q_constant,
            "q_adaptive": self.q_adaptive,
            "width_threshold": self.width_threshold,
            "market_counts": self.market_counts,
            "min_market_rows": self.min_market_rows,
            "feature_builder": self.fb.to_dict(),            
        }
        (out / "intervals.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


    @classmethod
    def load(cls, out: Path) -> Calibrator:
        m = json.loads((out / "intervals.json").read_text(encoding="utf-8"))
        return cls(
            FeatureBuilder.from_dict(m["feature_builder"]),
            lgb.Booster(model_file=str(out / "difficulty_lightgbm.txt")),
            m["levels"],
            m["q_constant"],
            m["q_adaptive"],
            m["width_threshold"],
            m["market_counts"],
            m["min_market_rows"],
        )

def calibrate(df: pl.DataFrame, cfg: dict[str, Any], seed: int) -> Calibrator:
    """Fit the difficulty model on one part of ``df`` and the band sizes on the rest."""
    c = cfg["intervals"]
    cal = np.random.default_rng(seed).random(df.height) < c["calib_frac"]
    fit_df, cal_df = df.filter(pl.Series(~cal)), df.filter(pl.Series(cal))

    fb = FeatureBuilder().fit(fit_df.drop(OWN_COLUMNS))
    params = {**lgb_params(cfg), **c["difficulty"], "seed": seed, "verbose": -1}
    rounds = params.pop("num_boost_round")
    x = np.hstack([fb.matrix(fit_df), fit_df["pred_log"].to_numpy()[:, None]])
    data = lgb.Dataset(
        x, 
        np.log(np.abs(fit_df["resid"].to_numpy()) + EPS),
        feature_name=[*fb.feature_names, "pred_log"],
        categorical_feature=fb.categorical_names,
    )
    booster = lgb.train(params, data, num_boost_round=rounds)

    counts = df["market"].value_counts()
    cb = Calibrator(
        fb, 
        booster,
        list(c["coverage_levels"]),
        {},
        {},
        0.0,
        dict(zip(counts["market"].to_list(), counts["count"].to_list(), strict=True)),
        c["min_market_rows"],
    )
    err, sig = np.abs(cal_df["resid"].to_numpy()), cb.difficulty(cal_df)
    for lv in cb.levels:
        k = f"{round(lv * 100)}"
        cb.q_constant[k] = conformal_q(err, lv)
        cb.q_adaptive[k] = conformal_q(err / sig, lv)
    k = f"{round(cb.levels[0] * 100)}"
    lo, hi = band(cal_df["pred_log"].to_numpy(), cb.q_adaptive[k] * sig)
    cb.width_threshold = float(
        np.quantile((hi - lo) / cal_df["pred"].to_numpy(), 1 - c["review_rate"])
    ) 
    return cb   

def cross_validate(df: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Out-of-fold ranges: each fold is calibrated on the other folds only."""
    parts = []
    for f in sorted(df["fold"].unique().to_list()):
        cb = calibrate(df.filter(pl.col("fold") != f), cfg, cfg["seed"] + int(f))
        parts.append(cb.apply(df.filter(pl.col("fold") == f)))
    out = pl.concat(parts)
    return df.select("listing_id", "fold", "market", "room_type", "price", "pred").join(
        out, on="listing_id"
    )


## ------------------------------------------------------------------------------------- TABLES
def covered(lo: str, hi: str) -> pl.Expr:
    return (pl.col("price") >= pl.col(lo)) & (pl.col("price") <= pl.col(hi))

def coverage_table(oof: pl.DataFrame, levels: list[float]) -> pl.DataFrame:
    """Target vs achieved coverage and rnage width, overall and for easy / hard listings."""
    hard = oof["difficulty"] >= oof["difficulty"].quantile(2 / 3)
    easy = oof["difficulty"] <= oof["difficulty"].quantile(1 / 3)
    rows = []
    for lv in levels:
        k = f"{round(lv * 100)}"
        for method, lo, hi in (
            ("constant", f"lo_const_{k}", f"hi_const_{k}"),
            ("adaptive", f"lo_{k}", f"hi_{k}"),
        ):
            cov = oof.select(covered(lo, hi)).to_series().to_numpy()
            width = ((oof[hi] - oof[lo]) / oof["pred"]).to_numpy()
            rows.append(
                {
                    "target_pct": round(lv * 100),
                    "method": method,
                    "coverage_pct": round(100 * cov.mean(), 2),
                    "coverage_easy_third": round(100 * cov[easy.to_numpy()].mean(), 2),
                    "coverage_hard_third": round(100 * cov[hard.to_numpy()].mean(), 2),
                    "median_width_pct": round(100 * float(np.median(width)), 1),
                    "median_width_usd": round(float((oof[hi] - oof[lo]).median()), 2), 
                }
            )
        return pl.DataFrame(rows)


def routing_table(oof: pl.DataFrame, k: str) -> pl.DataFrame:
    """Accuracy of the listings priced automatically vs those sent to a person."""
    ape = ((oof["pred"] - oof["price"]).abs() / oof["price"]).to_numpy()
    big = ape > 0.5
    rows = []
    for name, mask in (
        ("priced automatically", ~oof["review"].to_numpy()),
        ("sent to review", oof["review"].to_numpy()),
        ("all listings", np.ones(oof.height, dtype=bool)),
    ):
        part = oof.filter(pl.Series(mask))
        m = price_metrics(part["price"].to_numpy(), part["pred"].to_numpy())
        rows.append(
            {
                "group": name,
                "rows": int(mask.sum()),
                "share_pct": round(100 * mask.mean(), 1),
                "mdape_pct": m["mdape_pct"],
                "within_10_pct": m["within_10_pct"],
                "within_20_pct": m["within_20_pct"],
                "errors_over_50pct_caught": round(100 * (big & mask).sum() / max(big.sum(), 1), 1),
                f"coverage_{k}_pct": round(
                    100 * part.select(covered(f"lo_{k}", f"hi_{k}")).to_series().mean(), 2
                ),
            }
        )
    return pl.DataFrame(rows)

def by_market(oof: pl.DataFrame, k: str) -> pl.DataFrame:
    return (
        oof.with_columns(
            covered(f"lo_const_{k}", f"hi_const_{k}").alias("c_const"),
            covered(f"lo_{k}", f"hi_{k}").alias("c_adapt"),
            ((pl.col(f"hi_{k}") - pl.col(f"lo_{k}")) / pl.col("pred")).alias("w"),
        )
        .group_by("market")
        .agg(
            pl.len().alias("rows"),
            (100 * pl.col("c_const").mean()).round(1).alias(f"coverage_constant_{k}"),
            (100 * pl.col("c_adapt").mean()).round(1).alias(f"coverage_adaptive_{k}"),
            (100 * pl.col("w").median()).round(1).alias("median_within_pct"),
            (100 * pl.col("review").mean()).round(1).alias("review_pct"),
        )
        .sort("rows", descending=True)
    )

def error_by_width(oof: pl.DataFrame) -> pl.DataFrame:
    """Do wider ranges really mean bigger errors? Actual error per decile of range width."""
    return (
        oof.with_columns(
            (pl.col("rel_width").rank("ordinal") * 10 // (oof.height + 1) + 1).alias(
                "width_decile"
            ),
            ((pl.col("pred") - pl.col("price")).abs() / pl.col("price") * 100).alias("ape"),
        )
        .group_by("width_decile")
        .agg(
            pl.len().alias("rows"),
            (100 * pl.col("rel_width").median()).round(1).alias("median_width_pct"),
            pl.col("ape").median().round(2).alias("median_actual_error_pct"),
        )
        .sort("width_decile")
    )


## ------------------------------------------------------------------------------------- REPORTS
def plot(err: pl.DataFrame, mk: pl.DataFrame, k: str, target: float, path: Path) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    a.plot(err["width_decile"], err["median_actual_error_pct"], "o-", color="#3b6ea8")
    a.set(
        xlabel = "Range width decile (1 = narrowest)",
        ylabel = "Median actual error %",
        title = "Wider range = bigger real errors",
    )
    m = mk.filter(pl.col("rows") >= 30)
    y = np.arange(m.height)
    b.barh(y - 0.2, m[f"coverage_constant_{k}"], 0.4, label="constant", color="#bbbbbb")
    b.barh(y + 0.2, m[f"coverage_adaptive_{k}"], 0.4, label="adaptive", color="#3b6ea8")
    b.axvline(target * 100, color = "black", ls="--", lw=1, label="target")
    b.set_yticks(y, m["market"].to_list())
    b.set(xlabel=f"Coverage of the {k}% range", title="Coverage by market", xlim=(50, 100))
    b.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"figures/{path.name}"

def write_report(path: Path, s: dict[str, Any], t: dict[str, pl.DataFrame], fig: str) -> None:
    k = s["route_level"]
    lines = [
        "# Price ranges and human review",
        "_Generated by `python -m src.intervals`. Do not edit by hand._",
        "",
        f"Point model: **{s['source_model']}** (Phase 7 best). {s['rows']} listings; "
        "every range is out-of-fold (sized on the other folds, measured on the held-out fold).",
        "",
        "## 1. Coverage: do the ranges contain the true price as often as promised?",
        md(t["coverage"]),
        "",
        "- `constant`: same +/- band (log scale) for everyone. `adaptive`: band scaled by the "
        "difficulty model.",
        "- `coverage_eay_third` / `coverage_hard_third`: listings the difficulty model rated "
        "easiest / hardest. A good range keeps both near the target.",
        "",
        f"![ranges]){fig})",
        "",
        "## 3. Human-review routing",
        f"Rule: send to a person if the {k}% range is wider than {s['width_threshold_pct']}% of "
        f"the predicted price (the widest {s['review_rate_pct']}% in calibration), or the market "
        f"has fewer than {s['min_market_rows']} listings.",
        "",
        md(t["routing"]),
        "",
        md(t["reasons"]),
        "",
        "## 4. By market",
        md(t["by_market"]),
        "",
        "## 5. Notes",
        "- Coverage is marginal (on average over listings), the guarantee conformal prediction "
        "gives; per-market coverage can vary, as section 4 shows."
        "- The final calibrator (all rows) is in `models/intervals/` for the web application.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg["seed"])
    paths, c = cfg["paths"], cfg["intervals"]
    reports, preds_dir = Path(paths["reports"]), Path(paths["predictions"])
    df = load_inputs(cfg)
    best_file = preds_dir / "best_model.json"
    source = (
        json.loads(best_file.read_text(encoding="utf-8"))["best"] if best_file.exists() else
        "best"
    )
    log.info("Ranges for %d listings around: %s", df.height, source)

    oof = cross_validate(df, cfg)
    k = f"{round(c["coverage_levels"][0] * 100)}"
    tables = {
        "coverage": coverage_table(oof, c["coverage_levels"]),
        "error_by_width": error_by_width(oof),
        "routing": routing_table(oof, k),
        "reasons": oof.filter(pl.col("review"))
        .group_by("reason")
        .len("rows")
        .sort("rows", descending=True),
        "by_market": by_market(oof, k),
    }
    fig = plot(
        tables["error_by_width"],
        tables["by_market"],
        k,
        c["coverage_levels"][0],
        reports / "figures" / "intervals.png",
    )

    final = calibrate(df, cfg, cfg["seed"])
    final.save(Path(paths["models"]) / "intervals", source)
    oof.write_parquet(preds_dir / "intervals_oof.parquet")

    route = {r["group"]: r for r in tables["routing"].to_dicts()}
    cov = {(r["target_pct"], r["method"]): r for r in tables["coverage"].to_dicts()}
    summary = {
        "rows": oof.height,
        "source_model": source,
        "route_level": k,
        "review_rate_pct": round(100 * c["review_rate"]),
        "min_market_rows": c["min_market_rows"],
        "width_threshold_pct": round(100 * final.width_threshold, 1),
        "coverage": tables["coverage"].to_dicts(),
        "flagged_pct": route["sent to review"]["share_pct"],
        "auto_mdape_pct": route["priced automatically"]["mdape_pct"],
        "review_mdape_pct": route["sent to review"]["mdape_pct"],
        "all_mdape_pct": route["all listings"]["mdape_pct"],
        "auto_within_10_pct": route["priced automatically"]["within_10_pct"],
        "big_errors_caught_pct": route["sent to review"]["errors_over_50pct_caught"],
    }
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
        write_report(reports / "intervals.md", summary, tables, fig)
    (reports / "intervals_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow_run(cfg, "intervals") as mlflow:
        if mlflow is not None:
            mlflow.log_params({"source_model": source, "review_rate": c["review_rate"]})
            for (lv, method), r in cov.items():
                mlflow.log_metric(f"coverage_{method}_{lv}", r["coverage_pct"])
            for key in ("flagged_pct", "auto_mdape_pct", "big_errors_caught_pct"):
                mlflow.log_metric(key, summary[key])
            mlflow.log_artifact(str(reports / "intervals.md"))


    for (lv, method), r in cov.items():
        print(
            f"{lv}% range, {method:<8} coverage {r['coverage_pct']:>6}%  "
            f")easy {r['coverage_easy_third']}%, hard {r['coverage_hard_third']}%) "
            f"median width {r['median_width_pct']}%"
        )
    print(
        f"Sent to review: {summary['flagged_pct']}% of listings, catching "
        f"{summary['big_errors_caught_pct']}%, all {summary['all_mdape_pct']}%"
    )
    print(
        f"MdAPE: auto-priced {summary['auto_mdape_pct']}%, "
        f"review {summary['review_mdape_pct']}%, all {summary['all_mdape_pct']}%"
    )
    print(f"Report: {reports / 'intervals.md'}")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8a: price ranges and review routing")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    run(load_config(args.config))

if __name__ == "__main__":
    main()