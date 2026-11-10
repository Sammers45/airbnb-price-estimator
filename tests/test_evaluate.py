"""Phase 7 tests: statistics on known answers, and a full run on synthetic listings."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src import data as d
from src import evaluate as ev
from src.utils import load_config
from tests.test_data import make_raw


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = copy.deepcopy(load_config("config.yaml"))
    c["paths"] = {
        "raw": str(tmp_path / "raw.parquet"),
        "processed": str(tmp_path / "listings.parquet"),
        "reports": str(tmp_path / "reports"),
        "predictions": str(tmp_path / "predictions"),
        "models": str(tmp_path / "models"),
    }
    c["mlflow"]["enabled"] = False
    c["baseline"].update(num_boost_round=300, early_stopping_rounds=30)
    c["evaluate"].update(pca_components=4, bootstrap=300)
    return c


def write_inputs(cfg: dict, n: int = 400) -> pl.DataFrame:
    """Processed data + out-of-fold files. The text prediction is close to the true price."""
    df, _ = d.clean(make_raw(n), cfg, "all")
    df = df.with_columns(
        pl.Series(
            "fold", d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
        )
    )
    df.write_parquet(cfg["paths"]["processed"])
    rng = np.random.default_rng(3)
    preds = Path(cfg["paths"]["predictions"])
    preds.mkdir(parents=True, exist_ok=True)
    y = df["log_price"].to_numpy()
    base = df.select("listing_id", "fold", "price")
    base.with_columns(pl.Series("pred", np.expm1(y + rng.normal(0, 0.3, df.height)))).write_parquet(
        preds / "baseline_oof.parquet"
    )
    text_log = y + rng.normal(0, 0.05, df.height)  # the text "knows" the price
    base.with_columns(
        pl.Series("pred_log", text_log),
        pl.Series("pred", np.expm1(text_log)),
        pl.Series("text_vec", rng.normal(size=(df.height, 12)).astype(np.float32)).cast(
            pl.List(pl.Float32)
        ),
    ).write_parquet(preds / "text_oof.parquet")
    has = df["has_image_emb"].to_numpy() == 1
    img_log = np.where(has, y + rng.normal(0, 0.5, df.height), np.nan)
    base.select("listing_id").with_columns(
        pl.Series("has_image", has.astype(np.int8)),
        pl.Series("pred_log", img_log),
        pl.Series("pred", np.expm1(img_log)),
        *[
            pl.Series(f"img_{p}", np.where(has, rng.normal(0, 0.05, df.height), np.nan))
            for p in ("luxury", "cramped")
        ],
    ).write_parquet(preds / "image_oof.parquet")
    for arch in ("concat", "attention"):
        base.with_columns(
            pl.Series("pred", np.expm1(y + rng.normal(0, 0.28, df.height)))
        ).write_parquet(preds / f"fusion_{arch}_oof.parquet")
    return df


# ---------------------------------------------------------------- statistics
def test_bootstrap_of_identical_models_shows_no_change():
    y = np.linspace(50, 200, 300)
    p = y * 1.1
    r = ev.paired_bootstrap(y, p, p, 200, 0)
    assert r["mdape_pct"]["change"] == 0 and r["mdape_pct"]["p"] == 1.0


def test_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 200, 1000)
    worse = y * np.exp(rng.normal(0, 0.4, 1000))
    better = y * np.exp(rng.normal(0, 0.1, 1000))
    r = ev.paired_bootstrap(y, worse, better, 500, 0)
    assert r["mdape_pct"]["change"] < 0 and r["mdape_pct"]["ci_high"] < 0
    assert r["mdape_pct"]["p"] < 0.01 and r["within_10_pct"]["change"] > 0
    assert ev.paired_ttest(y, worse, better) < 0.001


def test_holm_correction_known_values():
    assert ev.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ev.holm([0.5]) == [0.5]


def test_paired_ttest_identical_is_one():
    y = np.array([100.0, 120.0, 80.0])
    assert ev.paired_ttest(y, y * 1.1, y * 1.1) == 1.0


# ---------------------------------------------------------------- features
def test_load_inputs_explains_missing_files(cfg):
    write_inputs(cfg)
    (Path(cfg["paths"]["predictions"]) / "text_oof.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Phase 4"):
        ev.load_inputs(cfg)


def test_extra_features_fit_on_training_rows_only(cfg):
    write_inputs(cfg)
    df = ev.load_inputs(cfg)
    train_rows = np.arange(0, df.height, 2)
    x, names, state = ev.extra_features(df, train_rows, ["text", "image"], 4, 0)
    t = np.array(df["text_vec"].to_list())
    assert np.allclose(state["text_pca"]["mean"], t[train_rows].mean(axis=0), atol=1e-5)
    assert x.shape == (df.height, len(names))
    no_photo = df["has_image_emb"].to_numpy() == 0
    assert np.isnan(x[no_photo, names.index("img_pc1")]).all()
    assert (x[no_photo, names.index("has_image")] == 0).all()
    assert {"text_pred_log", "img_pred_log", "img_luxury"} <= set(names)


# ---------------------------------------------------------------- end to end
def test_run_ablation_significance_and_outputs(cfg, tmp_path):
    write_inputs(cfg)
    summary = ev.run(cfg)
    tests = {r["question"]: r for r in summary["tests"]}
    text = tests["Does text add to the table?"]
    assert text["mdape_change"] < 0 and text["verdict"] == "better"
    assert {"Fusion NN: concatenation", "LightGBM baseline (Phase 3)"} <= {
        r["model"] for r in summary["leaderboard"]
    }

    best = pl.read_parquet(tmp_path / "predictions" / "best_oof.parquet")
    assert best.height == 400 - 3 and best["pred"].is_finite().all()
    report = (tmp_path / "reports" / "evaluation.md").read_text(encoding="utf-8")
    for heading in ("## 1. Leaderboard", "## 2. Ablation", "## 3. Significance"):
        assert heading in report

    import lightgbm as lgb

    meta = json.loads(
        (tmp_path / "models" / "stacked" / "stacked_features.json").read_text(encoding="utf-8")
    )
    booster = lgb.Booster(model_file=str(tmp_path / "models" / "stacked" / "stacked_lightgbm.txt"))
    assert booster.num_feature() == len(meta["features"])
    if "image" in meta["inputs"]:
        assert len(meta["image_ridge"]["coef"]) == 512


def test_table_features_never_include_other_models_outputs(cfg):
    from src.baseline import FeatureBuilder

    write_inputs(cfg)
    df = ev.load_inputs(cfg).with_columns(
        pl.lit(4.0).alias("base_log"), pl.lit(0.3).alias("w_text")
    )
    names = FeatureBuilder().fit(df).feature_names
    leaked = [
        n for n in names if n.startswith(("img_", "text_", "w_")) or n in ("base_log", "has_image")
    ]
    assert leaked == []