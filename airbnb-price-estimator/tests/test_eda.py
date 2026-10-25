"""Phase 2 tests: each analysis on small hand-made data, then a full run on synthetic listings."""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from src import data as d
from src import eda
from src.utils import load_config
from tests.test_data import make_raw


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = copy.deepcopy(load_config("config.yaml"))
    c["paths"] = {
        "raw": str(tmp_path / "raw.parquet"),
        "processed": str(tmp_path / "listings.parquet"),
        "reports": str(tmp_path / "reports"),
    }
    return c


@pytest.fixture
def processed(cfg) -> pl.DataFrame:
    """Synthetic listings run through the real Phase 1 cleaning + folds, then saved."""
    df, _ = d.clean(make_raw(300), cfg, "all")
    folds = d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
    df = df.with_columns(pl.Series("fold", folds))
    df.write_parquet(cfg["paths"]["processed"])
    return df


def test_relative_price_is_zero_median_per_market():
    df = pl.DataFrame(
        {"market": ["A", "A", "A", "B", "B", "B"], "log_price": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]}
    )
    out = eda.add_relative_price(df)
    assert out["rel_log_price"].to_list() == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]


def test_distance_to_centre_haversine():
    # 1 degree of latitude is about 111.2 km.
    df = pl.DataFrame({"market": ["A"] * 3, "latitude": [0.0, 0.0, 1.0], "longitude": [0.0] * 3})
    out = eda.add_distance_to_centre(df)["dist_centre_km"].to_list()
    assert out[0] == pytest.approx(0.0) and out[2] == pytest.approx(111.2, abs=0.1)


def test_suspicious_prices_flags_only_the_extreme_listing():
    logp = [4.0, 4.1, 3.9, 4.05, 3.95, 4.02, 3.98, 9.0]
    df = eda.add_relative_price(
        pl.DataFrame(
            {
                "listing_id": [str(i) for i in range(8)],
                "market": ["A"] * 8,
                "log_price": logp,
                "price": np.expm1(logp),
            }
        )
    )
    flagged = eda.suspicious_prices(df, z=3.5)
    assert flagged["listing_id"].to_list() == ["7"]


def test_suspicious_prices_handles_identical_prices():
    df = eda.add_relative_price(
        pl.DataFrame(
            {
                "listing_id": ["1", "2"],
                "market": ["A", "A"],
                "log_price": [4.0, 4.0],
                "price": [54.6] * 2,
            }
        )
    )
    assert eda.suspicious_prices(df, z=3.5).height == 0


def test_spearman_with_price():
    df = pl.DataFrame(
        {
            "up": list(range(20)),
            "down": list(range(20, 0, -1)),
            "flat": [1] * 20,
            "y": [float(i) for i in range(20)],
        }
    )
    out = eda.spearman_with_price(df, ["up", "down", "flat"], "y")
    rho = dict(zip(out["feature"], out["spearman"], strict=True))
    assert rho == {"up": 1.0, "down": -1.0}  # constant column skipped


def test_amenity_lift_recovers_a_doubling():
    rel = [np.log(2)] * 20 + [0.0] * 20  # listings with a pool cost 2x
    df = pl.DataFrame({"amen_pool": [1] * 20 + [0] * 20, "rel_log_price": rel})
    out = eda.amenity_lift(df)
    assert out["amenity"][0] == "pool" and out["lift_pct"][0] == pytest.approx(100.0)


def test_linear_probe_finds_signal_and_not_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8))
    folds = np.arange(500) % 5
    signal = x @ rng.normal(size=8)
    assert eda.linear_probe(x, signal, folds, [0.1, 1.0, 10.0]) > 0.95
    assert eda.linear_probe(x, rng.normal(size=500), folds, [0.1, 1.0, 10.0]) < 0.05


def test_embedding_matrix_rows_are_unit_length():
    df = pl.DataFrame({"e": [[3.0, 4.0], [0.0, 0.0]]})
    x = eda.embedding_matrix(df, "e")
    assert x[0] == pytest.approx([0.6, 0.8]) and (x[1] == 0).all()


def test_numeric_feature_columns_excludes_target_and_ids(processed):
    cols = eda.numeric_feature_columns(eda.add_relative_price(processed))
    assert "accommodates" in cols
    assert not {"price", "log_price", "fold", "price_local"} & set(cols)
    assert not any(c.startswith(("amen_", "has_")) for c in cols)


def test_full_eda_run(processed, cfg, tmp_path):
    summary = eda.run(cfg)
    report = (tmp_path / "reports" / "eda.md").read_text(encoding="utf-8")
    for heading in ("## 1. Target", "## 5. Numeric", "## 9. Signal check"):
        assert heading in report
    for fig in ("eda_price_by_market", "eda_price_vs_guests", "eda_spearman", "eda_amenities"):
        assert (tmp_path / "reports" / "figures" / f"{fig}.png").exists()
    saved = json.loads((tmp_path / "reports" / "eda_summary.json").read_text(encoding="utf-8"))
    assert saved["target"]["n"] == processed.height
    probes = {r["input"] for r in saved["tables"]["probes"]}
    assert {"market only", "text embedding", "image embedding"} <= probes
    assert summary["target"]["skew_log_price"] < summary["target"]["skew_price"]