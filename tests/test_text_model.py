"""Phase 4 tests. A tiny DistilBERT is built on disk, so nothing is downloaded and
the whole file runs on a CPU in seconds."""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src import text_model as tm  # noqa: E402
from src.utils import load_config  # noqa: E402

WORDS = [
    "cheap",
    "budget",
    "basic",
    "luxury",
    "villa",
    "pool",
    "ocean",
    "view",
    "room",
    "flat",
    "near",
    "metro",
    "quiet",
    "big",
    "small",
    "the",
    "a",
    "with",
]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A 1-layer DistilBERT with a 23-token vocabulary, saved like a Hugging Face model."""
    d = tmp_path_factory.mktemp("tiny_distilbert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    config = transformers.DistilBertConfig(
        vocab_size=len(vocab),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    transformers.DistilBertModel(config).save_pretrained(d)
    return d


@pytest.fixture
def cfg(tmp_path, tiny_model_dir) -> dict:
    c = copy.deepcopy(load_config("config.yaml"))
    c["paths"] = {
        "raw": str(tmp_path / "raw.parquet"),
        "processed": str(tmp_path / "listings.parquet"),
        "reports": str(tmp_path / "reports"),
        "predictions": str(tmp_path / "predictions"),
        "models": str(tmp_path / "models"),
    }
    c["mlflow"]["enabled"] = False
    c["text_model"].update(
        model_name=str(tiny_model_dir),
        max_length=16,
        batch_size=16,
        epochs=10,
        learning_rate=1e-3,
        warmup_frac=0.0,
    )
    return c


def make_listings(n: int = 100, seed: int = 0) -> pl.DataFrame:
    """Listings whose price depends on the words in their text."""
    rng = np.random.default_rng(seed)
    luxury = rng.random(n) < 0.5
    text = [
        ("luxury villa with pool ocean view" if lx else "cheap basic room near metro")
        + " "
        + " ".join(rng.choice(["quiet", "big", "small", "flat"], 2))
        for lx in luxury
    ]
    price = np.where(luxury, 300.0, 50.0) * np.exp(rng.normal(0, 0.1, n))
    return pl.DataFrame(
        {
            "listing_id": [f"{i:04d}" for i in range(n)],
            "fold": np.arange(n) % 5,
            "market": np.where(np.arange(n) % 2 == 0, "A", "B"),
            "room_type": "Entire home/apt",
            "price": price,
            "log_price": np.log1p(price),
            "text": text,
        }
    )


# ---------------------------------------------------------------- pieces
def test_tokenize_replaces_empty_text(cfg):
    _, tok = tm.new_model(cfg)
    ids = tm.tokenize(tok, ["cheap room", "", "   "], 16)
    assert len(ids) == 3 and all(len(x) >= 3 for x in ids)  # [CLS] token(s) [SEP]


def test_batches_pad_to_longest_and_mask(cfg):
    ids = [[2, 5, 3], [2, 5, 6, 7, 3]]
    rows, x, m = next(tm.batches(ids, np.array([0, 1]), 8, pad_id=0))
    assert x.shape == (2, 5) and x[0, 3:].tolist() == [0, 0]
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_mean_pooling_ignores_padding(cfg):
    model, tok = tm.new_model(cfg)
    model.eval()
    short = tm.tokenize(tok, ["cheap room"], 16)[0]
    longer = tm.tokenize(tok, ["luxury villa with pool ocean view"], 16)[0]
    with torch.no_grad():
        _, x1, m1 = next(tm.batches([short], np.array([0]), 1, 0))
        alone = model.embed(x1, m1)
        _, x2, m2 = next(tm.batches([short, longer], np.array([0, 1]), 2, 0))
        padded = model.embed(x2, m2)[0:1]
    assert torch.allclose(alone, padded, atol=1e-5)


def test_fine_tune_learns_a_text_signal(cfg):
    df = make_listings(80)
    model, tok = tm.new_model(cfg)
    ids = tm.tokenize(tok, df["text"].to_list(), 16)
    y = (df["log_price"].to_numpy() - df["log_price"].mean()) / df["log_price"].std()
    rows = np.arange(df.height)
    model, _, _ = tm.fine_tune(model, ids, y, rows, None, 5, cfg, tok.pad_token_id, seed=0)
    pred, vec = tm.predict(model, ids, rows, cfg, tok.pad_token_id)
    assert np.corrcoef(pred, y)[0, 1] > 0.9
    assert vec.shape == (80, 32)


def test_predict_keeps_row_order(cfg):
    model, tok = tm.new_model(cfg)
    ids = tm.tokenize(tok, ["cheap", "luxury villa with pool ocean view", "room"], 16)
    rows = np.array([2, 0, 1])
    p_all, _ = tm.predict(model, ids, rows, cfg, tok.pad_token_id)
    p_one, _ = tm.predict(model, ids, np.array([0]), cfg, tok.pad_token_id)
    assert p_all[1] == pytest.approx(p_one[0], abs=1e-5)


# ---------------------------------------------------------------- end to end
def test_run_cross_validates_and_saves_everything(cfg, tmp_path):
    df = make_listings(200)
    df.write_parquet(cfg["paths"]["processed"])
    summary = tm.run(cfg)

    oof = pl.read_parquet(tmp_path / "predictions" / "text_oof.parquet")
    assert oof.height == 200 and oof["listing_id"].n_unique() == 200
    assert oof["text_vec"].list.len().unique().to_list() == [32]
    text_row = summary["overall"][-1]
    assert text_row["model"] == "Text (DistilBERT)" and text_row["mdape_pct"] < 25
    assert "## 1. Overall" in (tmp_path / "reports" / "text_model.md").read_text(encoding="utf-8")

    model, tok, meta = tm.load_final(tmp_path / "models" / "text_model")
    ids = tm.tokenize(tok, ["luxury villa with pool", "cheap basic room"], meta["max_length"])
    pred, _ = tm.predict(model, ids, np.array([0, 1]), cfg, tok.pad_token_id)
    usd = np.expm1(pred * meta["target_std"] + meta["target_mean"])
    assert usd[0] > usd[1]  # the final model ranks luxury above cheap


def test_report_includes_baseline_when_available(cfg, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    base = {
        "model": "LightGBM baseline",
        "mae_usd": 1.0,
        "mdape_pct": 1.0,
        "within_10_pct": 1.0,
        "within_20_pct": 1.0,
        "r2_log": 1.0,
    }
    (reports / "baseline_metrics.json").write_text(json.dumps({"overall": [base]}))
    make_listings(50).write_parquet(cfg["paths"]["processed"])
    cfg["text_model"]["train_final"] = False
    summary = tm.run(cfg)
    assert [r["model"] for r in summary["overall"]] == ["LightGBM baseline", "Text (DistilBERT)"]


def test_log_results_records_run_in_mlflow(cfg, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    make_listings(50).write_parquet(cfg["paths"]["processed"])
    cfg["text_model"]["train_final"] = False
    tm.run(cfg)
    monkeypatch.chdir(tmp_path)
    cfg["mlflow"]["enabled"] = True
    # A database per test: MLflow caches stores, so a shared relative path would mix runs.
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tm.log_results(cfg)
    import mlflow

    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    assert len(runs) == 1 and "metrics.cv_mdape_pct" in runs.columns
