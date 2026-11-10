"""Phase 7: ablation and significance tests

Usage:
    python -m src.evaluate

1. Stacked LightGBM ablation: the baseline LightGBM retrained with growing inputs
    tab -> tab + text -> tab + image -> tab + text + image
   Text enters as the text model's out-of-fold predicted price + 8 principal components
   of the text vector; the photo as its predicted price, the 12 prompt scores, 8 principal
   components and a has-photo flag. Principal components are fit on training rows only. 

2. Leaderboard of every model (all out-of-fold, same 5,555 listings, same folds).

3. Significance: paired bootstrap on per-listing errors (95% CI and p-value for the change
    in MdAPE, MAE, and within-10%), a paired t-test on absolute log errors, Holm correction 
    across comparisons, and fold-by-fold consistency.

4. Saves the bets model's out-of-fold predictions (Phase 8) and a final stacked model trained
   on all rows (Phase 9).

Stacking caveat: text/photo predictions are out-of-fold but come from models
trained on the other folds; this standard practice can make stacked scores very slightly optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from src.baseline import FeatureBuilder, drop_outliers, lgb_params
from src.data import md
from src.image_model import image_matrix
from src.metrics import metrics_by, price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)

FEATURE_SETS = {
    "tab": [],
    "tab+text": ["text"],
    "tab+image": ["image"],
    "tab+text+image": ["text", "image"],
}
EXTERNAL = {
    "LightGBM baseline (Phase 3)": "baseline_oof.parquet",
    "Text only (DistilBERT)": "text_oof.parquet",
    "Fusion NN: concatenation": "fusion_concat_oof.parquet",
    "Fusion NN: attention": "fusion_attention_oof.parquet",
}

## ----------------------------------------------------------------- INPUTS

def load_inputs(cfg: dict[str, Any]) -> pl.DataFrame:
    """Processed table + text/photo outputs of Phase 4-5, one row per listing."""
    preds = Path(cfg["paths"]["predictions"])
    for f, how in {"text_oof.parquet": "Phase 4 (Colab)", "image_oof.parquet": "Phase 5"}.items():
        if not (preds / f).exists():
            raise FileNotFoundError(f"data/predictions/{f} is missing: run {how}")


    df = pl.read_parquet(cfg["paths"]["processed"])
    text = pl.read_parquet(preds / "text_oof.parquet").select(
        "listing_id", pl.col("pred_log").alias("text_pred_log"), "text_vec"
    )
    img = pl.read_parquet(preds / "image_oof.parquet")
    img = img.select(
        "listing_id",
        pl.col("pred_log").alias("img_pred_log"),
        *[c for c in img.columns if c.startswith("img_")],
    )
    out = df.join(text, on="listing_id").join(img, on="listing_id").sort("listing_id")
    if out.height != df.height:
        raise ValueError(
            "Prediction files cover fewer listings than the processed data; rerun them."
        )
    return out

def extra_features(
    df: pl.DataFrame, train_rows: np.ndarray, inputs: list[str], k: int, seed: int
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Stacking features for all rows; principal components fit on ``train_rows`` only.
    
    Returns (matrix, column names, fitted state for serving).
    """
    cols, names, state = [], [], {}
    if "text" in inputs:
        t = np.array(df["text_vec"].to_list(), dtype=np.float32)
        pca = PCA(n_components=k, random_state=seed).fit(t[train_rows])
        cols += [df["text_pred_log"].to_numpy()[:, None], pca.transform(t)]
        names += ["text_pred_log", *[f"text_pc{i + 1}" for i in range(k)]]
        state["text_pca"] = {"mean": pca.mean_.tolist(), "components":
        pca.components_.tolist()}

    if "image" in inputs:
        x, has = image_matrix(df)
        fit_rows = train_rows[has[train_rows]]
        pca = PCA(n_components=k, random_state=seed).fit(x[fit_rows])
        pcs = np.where(has[:, None], pca.transform(x), np.nan)
        prompts = [c for c in df.columns if c.startswith("img_") and c != "img_pred_log"]
        cols += [
            df["img_pred_log"].to_numpy()[:, None],
            has[:, None].astype(float),
            pcs,
            df.select(prompts).to_numpy().astype(float),
        ]
        names += ["img_pred_log", "has_image", *[f"img_pc{i + 1}" for i in range(k)],
                *prompts]
        state["image_pca"] = {"mean": pca.mean_.tolist(), "components": pca.components_.tolist()}
        state["prompts"] = prompts
    if not cols:
        return np.zeros((df.height, 0)), [], state
    return np.hstack(cols).astype(np.float64), names, state

def fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    cats: list[str],
    cfg: dict,
    seed: int,
    rounds: int | None = None,
) -> lgb.Booster:
    """LightGBM with the baseline settings; early stopping on 10% of the 
    TRAINING rows."""
    b = cfg["baseline"]
    if rounds:
        data = lgb.Dataset(x, y, feature_name=names, categorical_feature=cats)
        return lgb.train(lgb_params(cfg), data, num_boost_round=rounds)
    inner = np.random.default_rng(seed).random(len(y)) < b["inner_valid_frac"]
    train = lgb.Dataset(
        x[~inner], y[~inner], feature_name=names, categorical_feature=cats,
        free_raw_data=False
    )
    valid = lgb.Dataset(x[inner], y[inner], reference=train)
    return lgb.train(
        lgb_params(cfg),
        train,
        num_boost_round=b["num_boost_round"],
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(b["early_stopping_rounds"], verbose=False)],
    )

def stacked_cv(
    df: pl.DataFrame, cfg: dict, inputs: list[str]
) -> tuple[np.ndarray, pl.DataFrame, list[int]]:
    """Out-of-fold log-price predictions for one feature set, plus feature importance."""
    k = cfg["evaluate"]["pca_components"]
    folds = df["fold"].to_numpy()
    pred = np.zeros(df.height)
    gains, best_iters = [], []
    for f in sorted(np.unique(folds)):
        train_df = drop_outliers(df.filter(pl.col("fold") != f), cfg)
        train_rows = np.where(
            np.isin(df["listing_id"].to_numpy(), train_df["listing_id"].to_numpy())
        )[0]
        valid_rows = np.where(folds == f)[0]
        fb = FeatureBuilder().fit(train_df)
        base = fb.matrix(df)
        extra, names, _ = extra_features(df, train_rows, inputs, k, cfg["seed"] + int(f))
        x = np.hstack([base, extra])
        all_names = fb.feature_names + names
        booster = fit_lgbm(
            x[train_rows],
            df["log_price"].to_numpy()[train_rows],
            all_names,
            fb.categorical_names,
            cfg,
            cfg["seed"] + int(f),
        )
        pred[valid_rows] = booster.predict(x[valid_rows], num_iteration=booster.best_iteration)
        gains.append(dict(zip(all_names, booster.feature_importance("gain").tolist(), strict=True)))
        best_iters.append(booster.best_iteration)
    imp = (
        pl.DataFrame(gains)
        .fill_null(0)
        .mean()
        .transpose(include_header=True, header_name="feature", column_names=["gain"])
    )
    imp = imp.with_columns((100 * pl.col("gain") / pl.col("gain").sum()).round(2).alias("gain_pct"))
    return pred, imp.sort("gain", descending=True).select("feature", "gain_pct"), best_iters

## ----------------------------------------------------------------------------------------- STATISTICS

def paired_bootstrap(
    y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    """Change from model A to model B (B - A) with 95% CI and two-sided p-value.
    
    Each bootstrap sample resamples listings and scores BOTH models on the same listings,
    so the comparison is paired. Negative MdAPE/MAE change = B is better.
    """
    ape_a, ape_b = np.abs(pa - y) / y, np.abs(pb - y) / y
    ae_a, ae_b = np.abs(pa - y), np.abs(pb - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)), dtype=np.int32)
    diffs = {
        "mdape_pct": 100 * (np.median(ape_b[idx], axis=1) - np.median(ape_a[idx], axis=1)),
        "mae_usd": ae_b[idx].mean(axis=1) - ae_a[idx].mean(axis=1),
        "within_10_pct": 100 
        * ((ape_b[idx] <= 0.1).mean(axis=1) - (ape_a[idx] <= 0.1).mean(axis=1)),
    }
    point = {
        "mdape_pct": 100 * (np.median(ape_b) - np.median(ape_a)),
        "mae_usd": ae_b.mean() - ae_a.mean(),
        "within_10_pct": 100 * ((ape_b <= 0.1).mean() - (ape_a <= 0.1).mean()),
    }
    out = {}
    for m, d in diffs.items():
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[m] = {
            "change": round(float(point[m]), 3),
            "ci_low": round(float(np.percentile(d, 2.5)), 3),
            "ci_high": round(float(np.percentile(d, 97.5)), 3),
            "p": float(min(1.0, max(p, 1.0 / n_boot))),
        }
    return out

def paired_ttest(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> float:
    """p-value of a paired t-test on per-listing absolute log errors."""
    ea = np.abs(np.log1p(pa) - np.log1p(y))
    eb = np.abs(np.log1p(pb) - np.log1p(y))
    if np.allclose(ea, eb):
        return 1.0
    return float(stats.ttest_rel(eb, ea).pvalue)

def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls false `wins` across manu comparisons)"""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()

def fold_wins(df: pl.DataFrame, a: str, b: str) -> str:
    """'k/5': folds where model B has a lower MdAPE than Model A."""
    wins, total = 0, 0
    for (_,), part in df.group_by("fold"):
        y = part["price"].to_numpy()
        ma = np.median(np.abs(part[a].to_numpy() - y) / y)
        mb = np.median(np.abs(part[b].to_numpy() - y) / y)
        wins += int(mb < ma)
        total += 1
    return f"{wins}/{total}"


def compare(df: pl.DataFrame, pairs: list[tuple[str, str, str]], cfg: dict) -> pl.DataFrame:
    """Significance table for (question, model A, model B) pairs present in ``df`` """
    e = cfg["evaluate"]
    y = df["price"].to_numpy()
    rows = []
    for i, (question, a, b) in enumerate(pairs):
        if a not in df.columns or b not in df.columns:
            continue
        r = paired_bootstrap(y, df[a].to_numpy(), df[b].to_numpy(), e["bootstrap"], cfg["seed"] + i)
        rows.append(
            {
                "question": question,
                "A": a,
                "B": b,
                "mdape_change": r["mdape_pct"]["change"],
                "mdape_95ci": f"[{r['mdape_pct']['ci_low']}, {r['mdape_pct']['ci_high']}]",
                "p_bootstrap": r['mdape_pct']["p"],
                "mae_change": round(r["mae_usd"]["change"], 2),
                "within10_change": round(r["within_10_pct"]["change"], 2),
                "p_ttest_logerr": paired_ttest(y, df[a].to_numpy(), df[b].to_numpy()),
                "folds_B_wins": fold_wins(df, a, b),
            }
        )
    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows)
    adj = holm(out["p_bootstrap"].to_list())
    alpha = e["alpha"]
    return out.with_columns(
        pl.Series("p_holm", adj),
        pl.when((pl.Series(adj) < alpha) & (pl.col("mdape_change") < 0))
        .then(pl.lit("better"))
        .when((pl.Series(adj) < alpha) & (pl.col("mdape_change") > 0))
        .then(pl.lit("worse"))
        .otherwise(pl.lit("no siginificant difference"))
        .alias("verdict"),
    ).with_columns(
        pl.col(c).map_elements(fmt_p, return_dtype=pl.String)
        for c in ("p_bootstrap", "p_ttest_logerr", "p_holm")
    )

def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def model_ci(y: np.ndarray, pred: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% bootstrap interval of a single model's MdAPE"""
    ape = np.abs(pred - y) / y
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)), dtype=np.int32)
    meds = 100 * np.median(ape[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

## ---------------------------------------------------------------------------------- FINAL MODEL

def train_final(df: pl.DataFrame, cfg: dict, inputs: list[str], rounds: int, out: Path) -> None:
    """Best stacked model on all rows + everything needed to rebuild its inputs (web app)."""
    k = cfg["evaluate"]["pca_components"]
    train_df = drop_outliers(df, cfg)
    rows = np.where(np.isin(df["listing_id"].to_numpy(), train_df["listing_id"].to_numpy()))[0]
    fb = FeatureBuilder().fit(train_df)
    extra, names, state = extra_features(df, rows, inputs, k, cfg["seed"])
    x = np.hstack([fb.matrix(df), extra])
    booster = fit_lgbm(
        x[rows],
        df["log_price"].to_numpy()[rows],
        fb.feature_names + names,
        fb.categorical_names,
        cfg, 
        cfg["seed"],
        rounds=rounds,
    )   
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "stacked_lightgbm.txt"))
    if "image" in inputs:
        xi, has = image_matrix(df)
        ridge = RidgeCV(alphas=[float(a) for a in cfg["image_model"]["probe_alphas"]])
        ridge.fit(xi[has], df["log_price"].to_numpy()[has])
        state['image_ridge'] = {"coef": ridge.coef_.tolist(), "intercept":
                                float(ridge.intercept_)}
    meta = {
        "inputs": inputs,
        "features": fb.feature_names + names,
        "rounds": rounds,
        "feature_builder": fb.to_dict(),
        **state,
    }
    (out / "stacked_features.json").write_text(json.dumps(meta), encoding="utf-8")


## ---------------------------------------------------------------------------------- REPORT
def plot_ablation(board: pl.DataFrame, path: Path) -> str:
    t = board.sort("mdape_pct", descending=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * t.height + 1.5))
    err = np.array([t["mdape_pct"] - t["ci_low"], t["ci_high"] - t["mdape_pct"]])
    ax.barh(t["model"].to_list(), t["mdape_pct"].to_numpy(), xerr=err, color="#3b6ea8", capsize=3)
    ax.set(xlabel="MdAPE % (lower is better), 95% bootstrap interval",
           title="All models, same listings",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return f"figures/{path.name}"

def write_report(path: Path, s: dict, t: dict[str, pl.DataFrame], fig: str) -> None:
    lines = [
        "# Evaluation: ablation and significance",
        "_Generated by `python -m src.evalaute`. Do not edit by hand._",
        f"\n{s['rows']} listings, Phase 1 folds; every number is out-of-fold, in USD, onn the same "
        f"listings. Bootstrap: {s['bootstrap']} paired resamples. Significance level "
        f"{s['alpha']} after Homl correction across {s['n_comparisons']} comparisons.",
        "\n## 1. Leaderboard",
        md(t["board"]),
        f"\n![leaderboard]({fig})",
        "\n## 2. Ablation (stacked lightGBM): what does each input add?",
        md(t["ablation"]),
        "\n## 3. Significance (B compared with A; negative MdAPE/MAE change = B better)",
        md(t["tests"]),
        "\n- `p_bootstrap`: paired bootstrap on per-listing errors (primary test)."
        "\n- `p_ttest_logerr`: paired t-test on per-listing absolute log errors (secondary)."
        "\n- `p_holm`: bootstrap p adjusted for the number of comparisons; `verdict` uses it."
        "\n- `folds_B_wins`: folds in which B had the lower MdAPE.",
        f"\n## 4. Best Model: {s['best']}",
        "Per market, compared with the Phase 3 baseline:",
        md(t["by_market"]),
        f"\n## 5. Feature importance of the best stacked model ({s['best_stacked']}, top 20)",
        md(t["importance"].head(20)),
        "\n## 6. Notes",
        "- Stacking: text/photo predictions are out-of-fold but come from models trained on the "
        "other folds (standard practice; can be very slightly optimistic).",
        f"- Final model for the web app: `{s['final_model']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg["seed"])
    e, paths = cfg["evaluate"], cfg["paths"]
    reports, preds_dir = Path(paths["reports"]), Path(paths["predictions"])
    reports.mkdir(parents=True, exist_ok=True)
    df = load_inputs(cfg)
    y = df["price"].to_numpy()
    preds = df.select("listing_id", "fold", "market", "room_type", "price")

    stacked, importances, iters = {}, {}, {}
    for name, inputs in FEATURE_SETS.items():
        p, imp, it = stacked_cv(df, cfg, inputs)
        label = f"Stacked LightGBM: {name}"
        stacked[label], importances[label], iters[label] = inputs, imp, it
        preds = preds.with_columns(pl.Series(label, np.expm1(p)))
        log.info("%s: MdAPE %.2f%%", label, price_metrics(y, np.expm1(p))["mdape_pct"])
    for label, f in EXTERNAL.items():
        path = preds_dir / f
        if path.exists():
            ext = pl.read_parquet(path).select("listing_id", pl.col("pred").alias(label))
            preds = preds.join(ext, on="listing_id", how="left")

    models = [
        c for c in preds.columns if c not in ("listing_id", "fold", "market", "room_type", "price")
    ]
    board_rows = []
    for i, m in enumerate(models):
        lo, hi = model_ci(y, preds[m].to_numpy(), e["bootstrap"], cfg["seed"] + i)
        board_rows.append(
            {
                "model": m,
                **price_metrics(y, preds[m].to_numpy()),
                "ci_low": round(lo, 2),
                "ci_high": round(hi, 2),
            }
        )
    board = pl.DataFrame(board_rows).sort("mdape_pct")
    base = "Stacked LightGBM: tab"
    ablation = board.filter(pl.col("model").str.starts_with("Stacked")).with_columns(
        (pl.col("mdape_pct") - board.filter(pl.col("model") == base)["mdape_pct"][0])
        .round(2)
        .alias("mdape_vs_tab")
    )
    ref = "LightGBM baseline (Phase 3)" if "LightGBM baseline (Phase 3)" in models else base
    best = board["model"][0]
    pairs = [
        ("Does text add to the table?", base, "Stacked LightGBM: tab+text"),
        (
            "Does the photo add on top of table + text?",
            "Stacked LightGBM: tab+text",
            "Stacked LightGBM: tab+text+image",
        ),
        ("Does the photo add to the table alone?", base, "Stacked LightGBM: tab+image"),
        ("Table + text + photo vs table", base, "Stacked LightGBM: tab+text+image"),
        ("Fusion NN (concatenation) vs baseline", ref, "Fusion NN concatenation"),
        ("Fusion NN (attention) vs baseline", ref, "Fusion NN: attention"),
        ("Attention vs concatenation (architecture)",
         "Fusion NN: concatenation",
         "Fusion NN:attention",
        ),
        (
            "Stacked (all inputs) vs fusion NN (concatenation)",
            "Fusion NN: concatenation",
            "Stacked LightGBM: tab+text+image",
        ),
    ]
    if best not in {b for _, _, b in pairs} and best != ref:
        pairs.append(("Best model vs baseline", ref, best))
    tests = compare(preds, pairs, cfg)

    best_stacked = ablation.sort("mdape_pct")["model"][0]
    by_market = (
        metrics_by(preds.select("market", "price", pl.col(ref).alias("pred")), "market")
        .select("market", "rows", pl.col("mdape_pct").alias("baseline_mdape"))
        .join(
            metrics_by(
                preds.select("market", "price", pl.col(best).alias("pred")), "market"
            ).select("market", pl.col("mdape_pct").alias("best_mdape")),
            on="market",
        )
        .with_columns((pl.col("best_mdape") - 
                       pl.col("baseline_mdape")).round(2).alias("change"))
        .sort("rows", descending=True)
    )

    preds.write_parquet(preds_dir / "all_models_oof.parquet")
    preds.select(
        "listing_id", "fold", "market", "room_type", "price", pl.col(best).alias("pred")
    ).write_parquet(preds_dir / "best_oof.parquet")
    final_dir = Path(paths["models"]) / "stacked"
    rounds = max(1, int(np.mean(iters[best_stacked])))
    train_final(df, cfg, stacked[best_stacked], rounds, final_dir)

    fig = plot_ablation(board, reports / "figures" / "eval_leaderboard.png")
    summary = {
        "rows": df.height,
        "bootstrap": e["bootstrap"],
        "alpha": e["alpha"],
        "n_comparisons": tests.height,
        "best": best,
        "best_stacked": best_stacked,
        "final_model": str(final_dir / "stacked_lightgbm.txt").replace("\\", "/"),
        "leaderboard": board.to_dicts(),
        "tests": tests.to_dicts(),
    }
    tables = {
        "board": board,
        "ablation": ablation.select(
            "model",
            "mae_usd",
            "mdape_pct",
            "within_10_pct",
            "within_20_pct",
            "r2_log",
            "mdape_vs_tab",
        ),
        "tests": tests,
        "by_market": by_market,
        "importance": importances[best_stacked],
    }
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
        write_report(reports / "evaluation.md", summary, tables, fig)
    (reports / "evaluation_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (preds_dir / "best_model.json").write_text(
        json.dumps({"best": best, "best_stacked": best_stacked}), encoding="utf-8"
    )

    with mlflow_run(cfg, "evaluation") as mlflow:
        if mlflow is not None:
            mlflow.log_params(
                {"bootstrap": e["bootstrap"], "pca_components": e["pca_components"],
                 "best": best}
            )
            for row in board.iter_rows(named=True):
                key = "".join(ch if ch.isalnum() else "_" for ch in row['model']).strip("_").lower()
                mlflow.log_metric(f"mdape_{key}", row["mdape_pct"])
            for p in (
                reports / "evaluation.md",
                reports / "evaluation_metrics.json",
                reports / fig,
            ):
                mlflow.log_artifact(str(p))

    for row in board.iter_rows(named=True):
        print(
            f"{row['model']:<36} MdAPE {row['mdape_pct']:>6}%  "
            f"within 10% {row['within_10_pct']:>6}%"
        )
    for row in tests.iter_rows(named=True):
        print(
            f"{row['question']:<52} change {row['mdape_change']:>6}  "
            f"p(Holm) {row['p_holm']:>6}  {row['verdict']}"
        )
    print(f"Report: {reports / 'evaluation.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 ablation and significance tests")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    run(load_config(args.config))


if __name__ == "__main__":
    main()