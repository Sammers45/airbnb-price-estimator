"""Phase 8b tests: SHAP adds up, explanations are readable, IG finds the informative input."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src import baseline as bl
from src import data as d
from src import explain as ex
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
    c["explain"].update(ig_rows=150, ig_steps=16)
    return c


def write_baseline(cfg: dict, n: int = 500) -> pl.DataFrame:
    """Processed listings + a saved Phase 3 model, as `python -m src.baseline` leaves them."""
    df, _ = d.clean(make_raw(n), cfg, "all")
    df = df.with_columns(
        pl.Series(
            "fold", d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
        )
    )
    df.write_parquet(cfg["paths"]["processed"])
    booster, fb = bl.train_final(df, cfg, 150)
    models = Path(cfg["paths"]["models"])
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / "baseline_lightgbm.txt"))
    (models / "baseline_features.json").write_text(json.dumps(fb.to_dict()), encoding="utf-8")
    return df


def test_shap_values_add_up_to_the_prediction(cfg):
    df = write_baseline(cfg)
    booster, fb = ex.load_baseline(Path(cfg["paths"]["models"]))
    x = fb.matrix(df)
    phi, base = ex.shap_values(booster, x)
    assert phi.shape == x.shape
    assert np.allclose(base + phi.sum(axis=1), booster.predict(x), atol=1e-6)


def test_explain_listing_is_readable_and_consistent(cfg):
    df = write_baseline(cfg)
    booster, fb = ex.load_baseline(Path(cfg["paths"]["models"]))
    out = ex.explain_listing(booster, fb, df[:1], top=4)
    assert len(out["reasons"]) == 4
    assert np.isclose(
        out["predicted_price"], np.expm1(booster.predict(fb.matrix(df[:1])))[0], atol=0.01
    )
    effects = [abs(r["effect_pct"]) for r in out["reasons"]]
    assert effects == sorted(effects, reverse=True)
    market = [
        r
        for r in ex.explain_listing(booster, fb, df[:1], top=50)["reasons"]
        if r["feature"] == "market"
    ]
    assert market and market[0]["value"] in fb.categories["market"]  # a name, not a code


def test_shap_table_ranks_the_strongest_feature_first(cfg):
    df = write_baseline(cfg)
    booster, fb = ex.load_baseline(Path(cfg["paths"]["models"]))
    x = fb.matrix(df)
    phi, _ = ex.shap_values(booster, x)
    t = ex.shap_table(phi, x, fb)
    assert t.height == len(fb.feature_names)
    assert t["mean_abs_shap"].is_sorted(descending=True)
    acc = t.filter(pl.col("feature") == "accommodates").row(0, named=True)
    assert acc["direction"] == "higher = pricier"  # more guests -> higher price in make_raw


def test_missing_baseline_explains_which_phase(cfg):
    with pytest.raises(FileNotFoundError, match="Phase 3"):
        ex.load_baseline(Path(cfg["paths"]["models"]))


def test_run_without_fusion_model_skips_ig(cfg, tmp_path):
    write_baseline(cfg)
    s = ex.run(cfg)
    assert s["ig"] is None and len(s["examples"]) == 3
    report = (tmp_path / "reports" / "explain.md").read_text(encoding="utf-8")
    assert "## 1. What drives the price" in report and "Skipped" in report


def test_integrated_gradients_credit_the_informative_input(cfg):
    pytest.importorskip("captum")
    from src import fusion as fu
    from tests.test_fusion import write_inputs

    cfg["fusion"].update(seeds=1, batch_size=64)
    write_inputs(cfg, 400)  # the text vector holds what the baseline gets wrong
    # Make the text the ONLY informative input: shuffle the table between listings and give
    # every photo the same vector and prompt scores.
    proc = pl.read_parquet(cfg["paths"]["processed"])
    keep = ["listing_id", "fold", "price", "log_price", "price_local", "currency"]
    perm = np.random.default_rng(0).permutation(proc.height)
    proc = proc.select(keep).hstack(proc.drop(keep)[perm])
    same = proc.filter(pl.col("has_image_emb") == 1)["image_embeddings"][0].to_list()
    vecs = [same if h == 1 else None for h in proc["has_image_emb"].to_list()]
    proc.with_columns(
        pl.Series("image_embeddings", vecs, dtype=proc.schema["image_embeddings"])
    ).write_parquet(cfg["paths"]["processed"])
    img = Path(cfg["paths"]["predictions"]) / "image_oof.parquet"
    pl.read_parquet(img).with_columns(pl.col("^img_.*$") * 0).write_parquet(img)
    df = fu.load_inputs(cfg)
    fu.train_final(
        df, cfg, "concat", list(fu.MODALITIES), 40, Path(cfg["paths"]["models"]) / "fusion_concat"
    )
    ig = ex.integrated_gradients(cfg)
    r = ig["shares"].row(0, named=True)
    assert abs(r["table_pct"] + r["text_pct"] + r["photo_pct"] - 100) < 0.5
    assert r["text_pct"] > max(r["table_pct"], r["photo_pct"])
    assert ig["delta_ratio"] < 0.05  # attributions add up to the output change
    no_photo = ig["shares"].filter(pl.col("listings") == "no photo").row(0, named=True)
    assert no_photo["photo_pct"] < 1  # a missing photo is masked, so it gets ~no credit