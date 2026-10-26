"""Phase 3 tests: metrics, leakage-safe features, anf full baseline run on synthetic data"""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from src import baseline as b   
from src import data as d
from src.metrics import metrics_by, price_metrics
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
    c["baseline"]["num_boost_round"] = 300
    c["baseline"]["early_stopping_rounds"] = 30

    return c

@pytest.fixture
def processed(cfg) -> pl.DataFrame:
    """Synthetic listings through the real Phase 1 cleaning + folds, saved to disk."""
    df, _ = d.clean(make_raw(400), cfg, "all")
    folds = d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
    df = df.with_columns(pl.Series("fold", folds))
    df.write_parquet(cfg["paths"]["processed"])
    return df

## ---------------------------------------------------------------------- METRICS
def test_price_metrics_known_values():
    m = price_metrics(np.array([100.0, 100.0, 100.0, 100.0]), np.array([105.0, 90.0, 130.0, 100.0]))
    assert m["mae_usd"] == 11.25
    assert m["mdape_pct"] == 7.5
    assert m["within_10_pct"] == 75.0
    assert m["within_20_pct"] == 75.0

def test_metrics_by_groups_and_skips_tiny_groups():
    df = pl.DataFrame({"market": ["A"] * 6 + ["B"] * 2, "price": [100.0] * 8, "pred": [110.0] * 8})
    out = metrics_by(df, "market")
    assert out["market"].to_list() == ["A"] and out["mae_usd"][0] == 10.0


## ----------------------------------------------------------------------FEATURES
def _frame(markets, rooms, lats, first_review):
    n = len(markets)
    return pl.DataFrame(
        {
            "listing_id": [str(i) for i in range(n)],
            "market": markets,
            "room_type": rooms,
            "latitude": lats,
            "longitude": [0.0] * n,
            "accommodates": [2.0] * n,
            "days_since_first_review": first_review,
            "price": [50.0] * n,
            "log_price": [3.9] * n,
            "price_local": [99.0] * n,
            "text_n_words": [10] * n,
            "has_image_emb": [1] * n,
        }
    )

def test_feature_builder_learns_from_training_rows_only():
    train = _frame(
        ["A", "A", "B"], ["Entire", "Private", "Entire"], [0.0, 0.0, 10.0], [5.0, None, 1.0]
    )
    valid = _frame(["A", "C"], ["Shared", "Entire"], [1.0, 0.0], [None, 3.0])
    fb = b.FeatureBuilder().fit(train)
    assert fb.categories["room_type"] == ["Entire", "Private"]
    x = fb.transform(valid)
    assert x["room_type"].to_list() == [None, 0.0]
    assert x["market"].to_list() == [0.0, None]
    assert x["dist_centre_km"][0] == pytest.approx(111.2, abs=0.2)
    assert x["dist_centre_km"][1] is None
    assert x["has_reviews"].to_list() == [0.0, 1.0]

def test_feature_builder_never_uses_target_text_or_iamge_columns():
    fb = b.FeatureBuilder().fit(_frame(["A"], ["Entire"], [0.0], [1.0]))
    leaked = {"price", "log_price", "price_local", "text_n_words", "has_image_emb", "fold"}
    assert not leaked & set(fb.feature_names)


def test_feature_builder_round_trips_through_json():
    df = _frame(["A", "B"], ["Entire", "Private"], [0.0, 1.0], [1.0, None])
    fb = b.FeatureBuilder().fit(df)
    clone = b.FeatureBuilder.from_dict(json.loads(json.dumps(fb.to_dict())))
    assert clone.transform(df).equals(fb.transform(df))

def test_naive_market_median_falls_back_for_unseen_market():
    train = pl.DataFrame({"market": ["A", "A", "B"], "price": [10.0, 30.0, 100.0]})
    valid = pl.DataFrame({"market": ["A", "Z"]})
    assert b.naive_market_median(train, valid).tolist() == [20.0, 30.0]

def test_drop_outliers_uses_training_quantiles(cfg):
    train = pl.DataFrame({"log_price": np.linspace(0, 10, 101)})
    kept = b.drop_outliers(train, cfg)["log_price"]
    assert kept.min() >= 0.1 - 1e-9 and kept.max() <= 9.9 + 1e-9 and kept.len() == 99


## ------------------------------------------------------------------------------- END to END
def test_baseline_run_end_to_end(processed, cfg, tmp_path):
    summary = b.run(cfg)
    oof = pl.read_parquet(tmp_path / "predictions" / "baseline_oof.parquet")
    assert oof.height == processed.height and oof["listing_id"].n_unique() == processed.height
    assert (oof["pred"] > 0).all()
    naive, lgbm = summary["overall"]
    assert lgbm["mae_usd"] < naive["mae_usd"]
    assert (tmp_path / "models" / "baseline_lightgbm.txt").exists()
    report = (tmp_path / "reports" / "baseline.md").read_text(encoding="utf-8")
    assert '## 1. Overall' in report and "LightGBM baseline" in report

def test_final_model_reloads_and_predicts(processed, cfg, tmp_path):
    import lightgbm as lgb

    b.run(cfg)
    booster = lgb.Booster(model_file=str(tmp_path / "models" / "baseline_lightgbm.txt"))
    fb = b.FeatureBuilder.from_dict(
        json.loads((tmp_path / "models" / "baseline_features.json").read_text(encoding="utf-8"))
    )
    pred = np.expm1(booster.predict(fb.matrix(processed.head(5))))
    assert pred.shape == (5,) and (pred > 0).all()

def test_baseline_logs_to_mlflow(processed, cfg, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    cfg["mlflow"]["enabled"] = True 
    b.run(cfg)
    import mlflow

    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    assert len(runs) == 1 and "metrics.cv_mae_usd" in runs.columns 