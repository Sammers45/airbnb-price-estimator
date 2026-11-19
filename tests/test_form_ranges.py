"""Phase 10 tests: form-only rebuild, honest ranges for it, and the service using them."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import polars as pl
import pytest

from src import form_ranges as fr
from src import serve as sv
from src.utils import load_config
from tests.test_fusion import write_inputs


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
    c["baseline"].update(num_boost_round=200, early_stopping_rounds=20)
    c["intervals"]["difficulty"]["num_boost_round"] = 50
    c["serve"]["min_market_rows"] = 20
    write_inputs(c, 800)
    preds = Path(c["paths"]["predictions"])
    pl.read_parquet(preds / "baseline_oof.parquet").write_parquet(preds / "best_oof.parquet")
    return c


def test_form_rows_keep_form_fields_and_use_typical_values_for_the_rest(cfg):
    df = pl.read_parquet(cfg["paths"]["processed"])
    form = sv.build_form(cfg, df)
    rows = fr.form_rows(df, form)
    market = next(iter(form["markets"]))
    real = df.filter(pl.col("market") == market).sort("listing_id")
    rebuilt = rows.filter(pl.col("market") == market).sort("listing_id")
    assert rebuilt["accommodates"].to_list() == real["accommodates"].to_list()  # on the form
    assert rebuilt["price"].to_list() == real["price"].to_list()  # the truth is kept alongside
    assert rebuilt["number_of_reviews"].n_unique() == 1  # not on the form -> market typical
    assert real["number_of_reviews"].n_unique() > 1


def test_form_only_ranges_are_honest_and_saved(cfg, tmp_path):
    s = fr.run(cfg)
    assert s["mdape_form_pct"] > s["mdape_full_pct"]  # less information, bigger errors
    assert abs(s["after"]["coverage_80_pct"] - 80) < 6
    report = (tmp_path / "reports" / "form_ranges.md").read_text(encoding="utf-8")
    assert "before and after the fix" in report
    meta = json.loads((tmp_path / "models" / "intervals_form" / "intervals.json").read_text())
    assert meta["source_model"] == "LightGBM, form-only input"
    leaked = {"pred_full", "price", "log_price", "fold"} & set(meta["feature_builder"]["numeric"])
    assert not leaked  # the difficulty model only sees what the form provides



def test_values_beyond_the_form_limits_are_capped():
    form = {"amenities": []}
    x = sv.listing_from_row({"market": "Porto", "minimum_nights": 1250.0, "accommodates": 0.0}, form)
    assert x.minimum_nights == 365 and x.accommodates == 1  # the form's own limits

def test_service_uses_form_ranges_for_web_requests(cfg):
    pytest.importorskip("torch")
    import lightgbm as lgb  # noqa: F401

    from src import baseline as bl
    from src import intervals as iv
    from src import monitor as mon

    df = pl.read_parquet(cfg["paths"]["processed"])
    models = Path(cfg["paths"]["models"])
    models.mkdir(parents=True, exist_ok=True)
    booster, fb = bl.train_final(df, cfg, 100)
    booster.save_model(str(models / "baseline_lightgbm.txt"))
    (models / "baseline_features.json").write_text(json.dumps(fb.to_dict()), encoding="utf-8")
    iv.run(cfg)
    fr.run(cfg)
    (models / "serving").mkdir(exist_ok=True)
    (models / "serving" / "form.json").write_text(json.dumps(sv.build_form(cfg)), encoding="utf-8")
    mon.build_reference(cfg)

    cfg["serve"].update(model="baseline", log_dir=str(Path(cfg["paths"]["reports"]) / "logs"))
    p = sv.Predictor(cfg)
    x = sv.Listing(market=next(iter(p.form["markets"])))
    assert p.predict(x)["range_basis"] == "form only (new listing)"
    known = df.filter(pl.col("market") == x.market).row(0, named=True)
    assert p.predict(x, known=known)["range_basis"] == "full listing data"