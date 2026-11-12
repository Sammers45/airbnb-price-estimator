"""Phase 8a tests: synthetic listings whose price errors grow with the number of guests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src import data as d
from src import intervals as iv
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
    c["intervals"]["difficulty"]["num_boost_round"] = 100
    return c


def write_inputs(cfg: dict, n: int = 1500) -> pl.DataFrame:
    """Processed table + a 'best model' whose error is small for 1-2 guests, large for 8-9."""
    df, _ = d.clean(make_raw(n), cfg, "all")
    df = df.with_columns(
        pl.Series(
            "fold", d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
        )
    )
    df.write_parquet(cfg["paths"]["processed"])
    rng = np.random.default_rng(7)
    scale = 0.03 * df["accommodates"].to_numpy()  # 3% per guest
    pred = np.expm1(df["log_price"].to_numpy() + rng.normal(0, 1, df.height) * scale)
    preds = Path(cfg["paths"]["predictions"])
    preds.mkdir(parents=True, exist_ok=True)
    df.select("listing_id", "fold", "market", "room_type", "price").with_columns(
        pl.Series("pred", pred)
    ).write_parquet(preds / "best_oof.parquet")
    (preds / "best_model.json").write_text(json.dumps({"best": "Synthetic model"}))
    return df


# ---------------------------------------------------------------- building blocks
def test_conformal_quantile_finite_sample_correction():
    s = np.arange(1, 10)  # 9 scores
    assert iv.conformal_q(s, 0.8) == 8  # ceil(10 * 0.8) = 8th smallest
    assert iv.conformal_q(s, 0.99) == 9  # never beyond the largest score


def test_band_is_symmetric_on_log_scale_and_never_negative():
    lo, hi = iv.band(np.log1p(np.array([100.0])), np.array([0.2]))
    assert np.isclose(np.log1p(hi) - np.log1p(100), 0.2)
    assert np.isclose(np.log1p(100) - np.log1p(lo), 0.2)
    lo, _ = iv.band(np.array([0.1]), np.array([5.0]))
    assert lo[0] == 0


def test_load_inputs_explains_missing_phase7(cfg):
    write_inputs(cfg, 200)
    (Path(cfg["paths"]["predictions"]) / "best_oof.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Phase 7"):
        iv.load_inputs(cfg)


# ---------------------------------------------------------------- behaviour
@pytest.fixture
def oof(cfg):
    write_inputs(cfg)
    return iv.cross_validate(iv.load_inputs(cfg), cfg)


def test_out_of_fold_coverage_matches_target(oof, cfg):
    t = iv.coverage_table(oof, cfg["intervals"]["coverage_levels"])
    for r in t.iter_rows(named=True):
        assert abs(r["coverage_pct"] - r["target_pct"]) < 5, r


def test_adaptive_ranges_follow_difficulty(oof, cfg):
    t = {(r["target_pct"], r["method"]): r for r in iv.coverage_table(oof, [0.8]).to_dicts()}
    const, adapt = t[(80, "constant")], t[(80, "adaptive")]
    # constant ranges under-cover the hard listings; adaptive ones keep them near 80%
    assert const["coverage_hard_third"] < adapt["coverage_hard_third"]
    assert abs(adapt["coverage_hard_third"] - 80) < abs(const["coverage_hard_third"] - 80)
    big = oof.filter(pl.col("difficulty") > pl.col("difficulty").median())
    small = oof.filter(pl.col("difficulty") <= pl.col("difficulty").median())
    assert big["rel_width"].median() > small["rel_width"].median()


def test_routing_sends_the_risky_listings_to_review(oof, cfg):
    r = {row["group"]: row for row in iv.routing_table(oof, "80").to_dicts()}
    assert 10 < r["sent to review"]["share_pct"] < 30  # review_rate 0.2
    assert r["sent to review"]["mdape_pct"] > r["priced automatically"]["mdape_pct"]
    assert r["priced automatically"]["mdape_pct"] < r["all listings"]["mdape_pct"]


def test_rare_market_always_goes_to_review(cfg):
    write_inputs(cfg, 600)
    df = iv.load_inputs(cfg)
    cb = iv.calibrate(df, cfg, 0)
    new = df.head(3).with_columns(pl.lit("Atlantis").alias("market"))
    out = cb.apply(new)
    assert out["review"].all() and (out["reason"] == "rare market").all()


def test_saved_calibrator_gives_identical_ranges(cfg, tmp_path):
    write_inputs(cfg, 600)
    df = iv.load_inputs(cfg)
    cb = iv.calibrate(df, cfg, 0)
    cb.save(tmp_path / "cal", "Synthetic model")
    again = iv.Calibrator.load(tmp_path / "cal").apply(df)
    assert np.allclose(again["lo_80"].to_numpy(), cb.apply(df)["lo_80"].to_numpy())


# ---------------------------------------------------------------- end to end
def test_run_writes_report_predictions_and_final_model(cfg, tmp_path):
    write_inputs(cfg, 800)
    s = iv.run(cfg)
    assert s["source_model"] == "Synthetic model"
    assert 0 < s["flagged_pct"] < 50
    report = (tmp_path / "reports" / "intervals.md").read_text(encoding="utf-8")
    for heading in ("## 1. Coverage", "## 3. Human-review routing"):
        assert heading in report
    oof = pl.read_parquet(tmp_path / "predictions" / "intervals_oof.parquet")
    assert {"lo_80", "hi_80", "lo_90", "review", "reason"} <= set(oof.columns)
    assert (oof["lo_80"] <= oof["pred"]).all() and (oof["pred"] <= oof["hi_80"]).all()
    assert (tmp_path / "models" / "intervals" / "difficulty_lightgbm.txt").exists()