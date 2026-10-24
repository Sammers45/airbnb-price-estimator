"""Phase 1 tests. They use a small synthetic dataset, so they run offline in CI."""

from __future__ import annotations

import copy
import sys
import types
from datetime import datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from src import data as d
from src.utils import load_config


def make_raw(n: int = 200) -> pl.DataFrame:
    """Fake listings shaped like MongoDB/airbnb_embeddings (nested structs, lists)."""
    rng = np.random.default_rng(0)
    pool = ["Wifi", "Kitchen", "Heating", "Washer", "TV", "Pool"]
    rows = []
    for i in range(n):
        bedrooms = int(rng.integers(0, 5))
        price = round(40 + 35 * bedrooms + float(rng.gamma(2, 15)), 2)
        rows.append({
            "_id": 1000 + i,  # int64 in the real data
            "listing_url": f"https://example.com/{i}",
            "name": f"Flat {i}",
            "summary": "Only $95 per night!" if i % 7 == 0 else "Bright room",
            "description": "Cleaning 80 euros" if i % 5 == 0 else "",
            "property_type": "Apartment",
            "room_type": ["Entire home/apt", "Private room"][i % 2],
            "bedrooms": bedrooms,
            "bathrooms": "1.5",
            "accommodates": bedrooms * 2 + 1,
            "price": f"${price:,.2f}",
            "weekly_price": price * 6,
            "cleaning_fee": 20.0,
            "number_of_reviews": int(rng.integers(0, 50)),
            "last_scraped": datetime(2019, 3, 1),
            "first_review": datetime(2019, 2, 1) if i % 3 else None,
            "last_review": None,
            "amenities": list(rng.choice(pool, 3, replace=False)),
            "images": {"picture_url": f"https://img.example.com/{i}.jpg"},
            "host": {"host_is_superhost": i % 2 == 0, "host_listings_count": 1 + i % 3},
            "address": {
                "country": ["United States", "Portugal"][i % 2],
                "market": ["New York", "Porto"][i % 2],
                "suburb": "Centro",
                "location": {"coordinates": [-8.6, 41.1]},
            },
            "review_scores": {"review_scores_rating": 90 + i % 10},
            "text_embeddings": rng.normal(size=1536).tolist(),
            "image_embeddings": rng.normal(size=512).tolist() if i % 9 else None,
        })  # fmt: skip
    rows[3]["price"] = None
    rows[5]["price"] = "free"
    rows[8]["price"] = "$0.00"
    return pl.DataFrame(rows, infer_schema_length=None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never download anything. If code falls back to `datasets`, the test fails."""

    def blocked(*args, **kwargs):
        raise AssertionError("tests must not download data")

    fake = types.ModuleType("datasets")
    fake.load_dataset = blocked
    monkeypatch.setitem(sys.modules, "datasets", fake)


@pytest.fixture
def raw() -> pl.DataFrame:
    return make_raw()


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = copy.deepcopy(load_config("config.yaml"))
    c["paths"] = {"raw": str(tmp_path / "raw.parquet"),
                  "processed": str(tmp_path / "listings.parquet"),
                  "reports": str(tmp_path / "reports")}  # fmt: skip
    c["data"]["min_market_rows"] = 50
    return c


def fake_urls(urls, n, timeout, threshold, seed):
    return {"tested": n, "failed": n, "fail_rate": 1.0, "use_provided_image_embeddings": True}


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "series, expected",
    [
        (pl.Series(["$1,200.00", "45", "free", None]), [1200.0, 45.0, None, None]),
        (pl.Series([80, 95]), [80.0, 95.0]),
        (pl.Series([Decimal("80.50")]), [80.5]),
        (pl.Series([{"$numberDecimal": "99.5"}]), [99.5]),
    ],
)
def test_to_float_handles_every_money_format(series, expected):
    df = pl.DataFrame({"p": series})
    assert df.select(d.to_float(pl.col("p"), df.schema["p"]).alias("p"))["p"].to_list() == expected


def test_to_flag():
    df = pl.DataFrame({"s": ["t", "f", "maybe", None], "b": [True, False, None, True]})
    assert df.select(d.to_flag(pl.col("s"), pl.String).alias("s"))["s"].to_list() == [
        1.0,
        0.0,
        None,
        None,
    ]
    assert df.select(d.to_flag(pl.col("b"), pl.Boolean).alias("b"))["b"].to_list() == [
        1.0,
        0.0,
        None,
        1.0,
    ]


def test_strip_prices():
    df = pl.DataFrame({"t": ["Only $95 per night, 80 euros, €200, 300 TL, 50/night. 2 bedrooms."]})
    out = df.select(d.strip_prices(pl.col("t")))["t"][0]
    assert all(n not in out for n in ("95", "80", "200", "300", "50"))
    assert out.count(d.PRICE_TOKEN) == 5 and "2 bedrooms" in out


# ---------------------------------------------------------------- features
def test_flatten(raw):
    flat = d.flatten(raw)
    assert set(flat["market"]) == {"New York", "Porto"}
    assert flat["latitude"][0] == pytest.approx(41.1)
    assert flat["rs_rating"].null_count() == 0
    assert flat["host_is_superhost"].to_list()[:2] == [1.0, 0.0]


def test_flatten_survives_missing_nested_fields():
    flat = d.flatten(pl.DataFrame({"price": [10.0]}))
    assert flat["market"][0] is None and flat["latitude"][0] is None


def test_amenities():
    df = pl.DataFrame({"amenities": [["Wifi", "TV"], ["Wifi"], []]})
    vocab = d.amenity_vocabulary(df, 2)
    assert vocab == ["Wifi", "TV"]
    out = d.amenity_features(df, vocab)
    assert out["amenities_count"].to_list() == [2, 1, 0]
    assert out["amen_wifi"].to_list() == [1, 1, 0]


def test_embedding_ok_flags_bad_vectors():
    df = pl.DataFrame({"e": [[1.0, 2.0], [1.0], None, [float("nan"), 1.0]]})
    assert df.select(d.embedding_ok(df, "e", 2))["e"].to_list() == [1, 0, 0, 0]


def test_clean(raw, cfg):
    df, info = d.clean(raw, cfg, "all")
    assert df.height == raw.height - 3  # None, "free", $0.00
    assert df["price"].dtype == pl.Float64
    assert np.allclose(df["log_price"], np.log1p(df["price"]))
    assert not set(cfg["features"]["leakage_columns"]) & set(df.columns)
    assert not df["text"].str.contains(r"\$95").any()
    assert df["has_image_emb"].sum() < df.height
    assert df["days_since_first_review"].drop_nulls()[0] == 28
    assert "weekly_price" in info["leakage_removed"]


def test_to_usd_converts_with_config_rates(cfg):
    df = pl.DataFrame({"country": ["United States", "Portugal", "Turkey"], "price": [100.0] * 3})
    out = d.to_usd(df, cfg)
    assert out["currency"].to_list() == ["USD", "EUR", "TRY"]
    assert out["price_local"].to_list() == [100.0] * 3
    assert out["price"].to_list() == pytest.approx([100.0, 100 / 0.8933, 100 / 5.6765])


def test_to_usd_rejects_unknown_country(cfg):
    with pytest.raises(ValueError, match="Narnia"):
        d.to_usd(pl.DataFrame({"country": ["Narnia"], "price": [10.0]}), cfg)


def test_clean_outputs_usd(raw, cfg):
    df, _ = d.clean(raw, cfg, "Porto")
    assert set(df["currency"]) == {"EUR"}
    assert np.allclose(df["price"], df["price_local"] / 0.8933)
    assert np.allclose(df["log_price"], np.log1p(df["price"]))


def test_clean_market_filter(raw, cfg):
    assert set(d.clean(raw, cfg, "Porto")[0]["market"]) == {"Porto"}
    with pytest.raises(ValueError):
        d.clean(raw, cfg, "Atlantis")
    with pytest.raises(ValueError):
        d.clean(raw, cfg, None)


# ---------------------------------------------------------------- splits and checks
def test_folds_balanced_reproducible_stratified():
    y = np.random.default_rng(0).normal(4.5, 0.6, 1000)
    a, b = d.make_folds(y, 5, 10, 42), d.make_folds(y, 5, 10, 42)
    assert (a == b).all() and np.bincount(a).min() >= 195
    deciles = np.argsort(np.argsort(y)) * 10 // 1000
    for k in range(5):  # every fold sees every price decile
        assert len(set(deciles[a == k])) == 10


def test_folds_with_markets_and_tiny_data():
    groups = np.array(["A"] * 20 + ["B"] * 3)
    folds = d.make_folds(np.linspace(3, 6, 23), 5, 10, 0, groups)
    assert set(folds) == set(range(5))


def test_folds_handle_tiny_markets_without_warning(recwarn):
    # Like the real data: big markets plus a few 1-2 row markets.
    groups = np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 2 + ["D"] * 1 + ["E"] * 2)
    y = np.random.default_rng(1).normal(4.5, 0.6, len(groups))
    folds = d.make_folds(y, 5, 10, 0, groups)
    assert set(folds) == set(range(5))
    assert not [w for w in recwarn if "least populated" in str(w.message)]


def test_outlier_bounds():
    assert d.outlier_bounds(np.arange(101.0), 0.05, 0.95) == (5.0, 95.0)


def test_url_check_counts_missing_as_failed():
    res = d.check_image_urls(["ok", "ok", None, "bad"], 4, 1, 0.2, 0,
                             fetch=lambda u, t: u == "ok")  # fmt: skip
    assert res["failed"] == 2 and res["use_provided_image_embeddings"]


def test_validate_rejects_bad_data(cfg):
    good = pl.DataFrame({"listing_id": ["1", "2"], "price": [50.0, 80.0],
                         "log_price": np.log1p([50.0, 80.0])})  # fmt: skip
    d.validate(good, cfg)
    for bad in (
        good.with_columns(pl.lit("1").alias("listing_id")),
        good.with_columns(pl.Series("price", [50.0, -1.0])),
        good.with_columns(pl.lit(1.0).alias("weekly_price")),
        good.with_columns(pl.Series("fold", [0, 0])),
    ):
        with pytest.raises(ValueError):
            d.validate(bad, cfg)


# ---------------------------------------------------------------- end to end
def test_audit_then_build(raw, cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        d.run_build(cfg, raw, "all")
    audit = d.run_audit(cfg, raw, url_check=fake_urls)
    assert audit["recommended_market"] in {"New York", "Porto"}
    assert "TBD" in (tmp_path / "reports" / "data_audit.md").read_text(encoding="utf-8")

    df = d.run_build(cfg, raw, "all")
    saved = pl.read_parquet(cfg["paths"]["processed"])
    assert saved.height == df.height and set(saved["fold"]) == set(range(5))
    assert saved["text_embeddings"].list.len()[0] == 1536
    report = (tmp_path / "reports" / "data_audit.md").read_text(encoding="utf-8")
    assert "Chosen market: **all**" in report and "TBD" not in report
    assert (tmp_path / "reports" / "figures" / "price_processed.png").exists()


# ---------------------------------------------------------------- loading
def test_download_reads_hf_parquet_with_polars(monkeypatch):
    seen = {}

    def fake_read(url):
        seen["url"] = url
        return pl.DataFrame({"price": [80]})

    monkeypatch.setattr(d.pl, "read_parquet", fake_read)
    assert d.download("MongoDB/airbnb_embeddings").height == 1
    assert seen["url"] == "hf://datasets/MongoDB/airbnb_embeddings@~parquet/**/*.parquet"


def test_download_falls_back_to_datasets(monkeypatch):
    def broken(url):
        raise OSError("no network")

    class FakeDataset:
        def to_polars(self):
            return pl.DataFrame({"price": [80, 90]})

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, split: FakeDataset()
    monkeypatch.setattr(d.pl, "read_parquet", broken)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert d.download("MongoDB/airbnb_embeddings").height == 2


def test_load_raw_downloads_once_then_uses_cache(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(d, "download", lambda name: calls.append(name) or make_raw(10))
    first, second = d.load_raw(cfg), d.load_raw(cfg)
    assert calls == ["MongoDB/airbnb_embeddings"]
    assert first.height == second.height == 10