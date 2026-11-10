"""Phase 6 tests (offline): synthetic listings whose text holds what the baseline misses."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from src import data as d  # noqa: E402
from src import fusion as fu  # noqa: E402
from src.utils import load_config  # noqa: E402
from tests.test_data import make_raw  # noqa: E402


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
    c["fusion"].update(seeds=1, max_epochs=60, patience=10, batch_size=64)
    return c


def write_inputs(cfg: dict, n: int = 400) -> pl.DataFrame:
    """Processed table + baseline/text/image out-of-fold files.

    The baseline is off by a random factor per listing; the text vector contains that
    factor, so a fusion model that uses the text can correct the baseline.
    """
    df, _ = d.clean(make_raw(n), cfg, "all")
    df = df.with_columns(
        pl.Series(
            "fold", d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
        )
    )
    df.write_parquet(cfg["paths"]["processed"])
    rng = np.random.default_rng(1)
    err = rng.normal(0, 0.3, df.height)  # what the baseline gets wrong (log scale)
    preds = cfg["paths"]["predictions"]
    Path(preds).mkdir(parents=True, exist_ok=True)
    df.select("listing_id", "fold", "price").with_columns(
        (pl.col("price") * np.exp(-err)).alias("pred")
    ).write_parquet(f"{preds}/baseline_oof.parquet")
    text_vec = np.column_stack([err, rng.normal(size=(df.height, 15))]).astype(np.float32)
    df.select("listing_id", "fold", "price").with_columns(
        pl.Series("text_vec", text_vec).cast(pl.List(pl.Float32)),
        (pl.col("price") * np.exp(rng.normal(0, 0.5, df.height))).alias("pred"),
    ).write_parquet(f"{preds}/text_oof.parquet")
    has = df["has_image_emb"].to_numpy() == 1
    scores = np.where(has[:, None], rng.normal(0, 0.05, (df.height, 3)), np.nan)
    df.select("listing_id").with_columns(
        pl.Series("has_image", has.astype(np.int8)),
        *[pl.Series(f"img_{n}", scores[:, i]) for i, n in enumerate(["luxury", "cramped", "pool"])],
    ).write_parquet(f"{preds}/image_oof.parquet")
    return df


# ---------------------------------------------------------------- inputs
def test_load_inputs_names_missing_files(cfg):
    write_inputs(cfg)
    (Path(cfg["paths"]["predictions"]) / "text_oof.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="text_oof.parquet"):
        fu.load_inputs(cfg)


def test_load_inputs_rejects_out_of_date_files(cfg):
    write_inputs(cfg)
    path = f"{cfg['paths']['predictions']}/image_oof.parquet"
    pl.read_parquet(path).head(10).write_parquet(path)
    with pytest.raises(ValueError, match="out of date"):
        fu.load_inputs(cfg)


def test_load_inputs_joins_everything(cfg):
    df = write_inputs(cfg)
    joined = fu.load_inputs(cfg)
    assert joined.height == df.height
    assert {"base_log", "text_vec", "img_luxury"} <= set(joined.columns)


def test_tab_preprocessor_fits_on_training_rows_and_round_trips(cfg):
    write_inputs(cfg)
    df = fu.load_inputs(cfg)
    train, valid = df.filter(pl.col("fold") != 0), df.filter(pl.col("fold") == 0)
    tab = fu.TabPreprocessor().fit(train)
    num, cat = tab.transform(train)
    assert num.shape == (train.height, tab.n_numeric) and cat.shape[1] == len(tab.categorical)
    assert np.allclose(
        num[:, : len(tab.numeric)].mean(axis=0), 0, atol=1e-4
    )  # standardised on train
    assert np.isfinite(num).all() and (cat >= 0).all()
    valid = valid.with_columns(pl.lit("Atlantis").alias("market"))
    _, vcat = tab.transform(valid)
    assert (vcat[:, tab.categorical.index("market")] == 0).all()  # unseen -> 0
    assert not [c for c in tab.numeric if c.startswith("img_") or c in ("base_log", "has_image")]
    clone = fu.TabPreprocessor.from_dict(json.loads(json.dumps(tab.to_dict())))
    assert np.array_equal(clone.transform(train)[0], num)


# ---------------------------------------------------------------- model
@pytest.fixture
def arrays(cfg):
    write_inputs(cfg)
    df = fu.load_inputs(cfg)
    tab = fu.TabPreprocessor().fit(df)
    a = fu.build_arrays(
        df, tab, fu.prompt_stats(df), float(df["log_price"].mean()), float(df["log_price"].std())
    )
    return df, tab, a


@pytest.mark.parametrize("arch", ["concat", "attention"])
def test_residual_model_starts_exactly_at_the_baseline(cfg, arrays, arch):
    _, tab, a = arrays
    net = fu.new_net(cfg, arch, list(fu.MODALITIES), tab, a)
    pred, _ = fu.predict(net, a)
    assert np.allclose(pred, a.base.numpy(), atol=1e-6)


def test_attention_ignores_a_missing_photo(cfg, arrays):
    df, tab, a = arrays
    net = fu.new_net(cfg, "attention", list(fu.MODALITIES), tab, a)
    _, w = fu.predict(net, a)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    no_photo = df["has_image_emb"].to_numpy() == 0
    assert no_photo.any() and np.allclose(w[no_photo, 2], 0.0)
    assert (w[~no_photo, 2] > 0).all()


def test_model_needs_the_table_and_supports_subsets(cfg, arrays):
    _, tab, a = arrays
    with pytest.raises(ValueError):
        fu.new_net(cfg, "concat", ["text", "image"], tab, a)
    for arch in ("concat", "attention"):
        net = fu.new_net(cfg, arch, ["tab", "text"], tab, a)
        pred, w = fu.predict(net, a)
        assert pred.shape == (a.y.shape[0],)
        assert w is None if arch == "concat" else w.shape == (a.y.shape[0], 2)


# ---------------------------------------------------------------- end to end
def test_run_beats_baseline_saves_outputs_and_final_models(cfg, tmp_path):
    write_inputs(cfg)
    summary = fu.run(cfg, ["concat", "attention"], list(fu.MODALITIES))
    comp = {r["model"]: r for r in summary["comparison"]}
    base = comp["LightGBM baseline"]["mdape_pct"]
    for name in ("Fusion: concatenation", "Fusion: attention"):
        assert comp[name]["mdape_pct"] < base, (name, comp[name]["mdape_pct"], base)
        assert comp[name]["mdape_vs_baseline"] < 0

    att = pl.read_parquet(tmp_path / "predictions" / "fusion_attention_oof.parquet")
    assert {"w_tab", "w_text", "w_image"} <= set(att.columns)
    assert att["pred"].is_finite().all()
    report = (tmp_path / "reports" / "fusion.md").read_text(encoding="utf-8")
    assert "## 1. Comparison" in report and "What the attention model relied on" in report

    meta = json.loads(
        (tmp_path / "models" / "fusion_attention" / "fusion.json").read_text(encoding="utf-8")
    )
    tab = fu.TabPreprocessor.from_dict(meta["tab"])
    df = fu.load_inputs(cfg)
    a = fu.build_arrays(
        df,
        tab,
        (
            np.array(meta["prompt_mean"], dtype=np.float32),
            np.array(meta["prompt_std"], dtype=np.float32),
        ),
        meta["target_mean"],
        meta["target_std"],
    )
    net = fu.new_net(cfg, meta["arch"], meta["modalities"], tab, a)
    net.load_state_dict(torch.load(tmp_path / "models" / "fusion_attention" / "model.pt"))
    pred, _ = fu.predict(net, a)
    assert np.isfinite(pred).all()


def test_modality_subset_run_writes_separate_files(cfg, tmp_path):
    write_inputs(cfg)
    cfg["fusion"]["max_epochs"] = 5
    fu.run(cfg, ["concat"], ["tab", "text"])
    assert (tmp_path / "predictions" / "fusion_concat_tab_text_oof.parquet").exists()
    assert (tmp_path / "reports" / "fusion_tab_text.md").exists()
    assert not (
        tmp_path / "models" / "fusion_concat"
    ).exists()  # final models only for the full run