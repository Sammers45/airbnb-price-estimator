"""Phase 5 tests: offline, using a fake encoder and a tiny CLIP model built on disk."""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from src import data as d
from src import image_model as im
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
    c["image_model"]["verify_photos"] = 5
    return c


@pytest.fixture
def processed(cfg) -> pl.DataFrame:
    """Synthetic listings through the real Phase 1 cleaning + folds (every 9th has no photo)."""
    df, _ = d.clean(make_raw(300), cfg, "all")
    folds = d.make_folds(df["log_price"].to_numpy(), 5, 10, 0, df["market"].to_numpy())
    df = df.with_columns(pl.Series("fold", folds))
    df.write_parquet(cfg["paths"]["processed"])
    return df


class FakeEncoder:
    """Images are dataset rows (looked up by URL); texts get fixed random vectors."""

    def __init__(self, by_url: dict[str, np.ndarray], noise: float = 0.0) -> None:
        self.by_url, self.noise = by_url, noise
        self.rng = np.random.default_rng(0)

    def images(self, images):
        out = np.stack([self.by_url[i] for i in images])
        return out + self.noise * self.rng.normal(size=out.shape)

    def texts(self, texts):
        return np.random.default_rng(len(texts)).normal(size=(len(texts), 512))


def fake_encoder(df: pl.DataFrame, noise: float = 0.0) -> FakeEncoder:
    rows = df.filter(pl.col("has_image_emb") == 1)
    vecs = {
        u: np.array(v) for u, v in zip(rows["picture_url"], rows["image_embeddings"], strict=True)
    }
    return FakeEncoder(vecs, noise)


def fetch_url(url, timeout):
    return url  # the "photo" is its URL; FakeEncoder looks it up


# ---------------------------------------------------------------- pieces
def test_unit_rows_have_length_one_and_zero_stays_zero():
    x = im.unit(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert x[0] == pytest.approx([0.6, 0.8]) and (x[1] == 0).all()


def test_image_matrix_masks_missing_photos(processed):
    x, has = im.image_matrix(processed)
    assert x.shape == (processed.height, 512)
    assert has.sum() == processed["has_image_emb"].sum() < processed.height
    assert np.allclose(np.linalg.norm(x[has], axis=1), 1.0) and (x[~has] == 0).all()


def test_prompt_scores_are_relative_to_each_photo():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    prompts = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = im.prompt_scores(x, prompts, ["a", "b"])
    assert s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0]
    assert np.allclose(s.mean(axis=1), 0.0)


# ---------------------------------------------------------------- verify
def test_verify_matches_when_encoder_reproduces_dataset_vectors(processed, cfg, tmp_path):
    result = im.verify(cfg, encoder=fake_encoder(processed), fetch=fetch_url)
    assert result["photos_compared"] == 5 and result["matches"]
    assert result["cosine_median"] == pytest.approx(1.0, abs=1e-4)
    saved = json.loads((tmp_path / "models" / "clip_prompts.json").read_text(encoding="utf-8"))
    assert list(saved["prompts"]) == list(cfg["image_model"]["prompts"])
    assert np.array(saved["vectors"]).shape == (len(cfg["image_model"]["prompts"]), 512)


def test_verify_flags_a_different_model(processed, cfg):
    result = im.verify(cfg, encoder=fake_encoder(processed, noise=5.0), fetch=fetch_url)
    assert not result["matches"]


def test_verify_skips_broken_links(processed, cfg):
    broken = iter([None, None, None])

    def flaky(url, timeout):
        return next(broken, url)

    result = im.verify(cfg, encoder=fake_encoder(processed), fetch=flaky)
    assert result["photos_tried"] == 8 and result["photos_compared"] == 5


# ---------------------------------------------------------------- train
def test_run_end_to_end(processed, cfg, tmp_path):
    im.verify(cfg, encoder=fake_encoder(processed), fetch=fetch_url)
    base = processed.select("listing_id", pl.col("price").alias("pred"))  # a perfect "baseline"
    (tmp_path / "predictions").mkdir(parents=True, exist_ok=True)
    base.write_parquet(tmp_path / "predictions" / "baseline_oof.parquet")

    summary = im.run(cfg)
    oof = pl.read_parquet(tmp_path / "predictions" / "image_oof.parquet")
    assert oof.height == processed.height
    missing = oof.filter(pl.col("has_image") == 0)
    assert missing["pred"].is_nan().all() and missing["img_luxury"].is_nan().all()
    with_img = oof.filter(pl.col("has_image") == 1)
    assert with_img["pred"].is_finite().all()
    assert {f"img_{n}" for n in cfg["image_model"]["prompts"]} <= set(oof.columns)
    assert [r["model"] for r in summary["overall"]] == [
        "LightGBM baseline (same listings)",
        "Image only (CLIP + ridge)",
    ]
    report = (tmp_path / "reports" / "image_model.md").read_text(encoding="utf-8")
    assert "## 1. Verification" in report and "match" in report and "## 4. Readable" in report
    table = report.split("## 4. Readable")[1]
    order = [line.split("|")[1].strip() for line in table.splitlines() if line.startswith("| ")][1:]
    corr = im.prompt_correlations(oof, list(cfg["image_model"]["prompts"]))["spearman_vs_price"]
    assert corr.abs().to_list() == sorted(corr.abs().to_list(), reverse=True)
    expected = im.prompt_correlations(oof, list(cfg["image_model"]["prompts"]))["prompt"].to_list()
    assert order == expected  # report keeps the strongest prompts first


def test_run_without_verify_still_trains(processed, cfg, tmp_path):
    summary = im.run(cfg)
    oof = pl.read_parquet(tmp_path / "predictions" / "image_oof.parquet")
    assert not any(c.startswith("img_") for c in oof.columns)
    assert summary["verify"] is None


# ---------------------------------------------------------------- real CLIP code path
@pytest.fixture(scope="module")
def tiny_clip_dir(tmp_path_factory):
    """A tiny random CLIP (8-d projection) with a real processor, saved like the hub model."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    d = tmp_path_factory.mktemp("tiny_clip")
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for ch in letters:
        vocab.setdefault(ch, len(vocab))
        vocab.setdefault(ch + "</w>", len(vocab))
    (d / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (d / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tok = transformers.CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"))
    imgp = transformers.CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    transformers.CLIPProcessor(image_processor=imgp, tokenizer=tok).save_pretrained(d)
    config = transformers.CLIPConfig(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 32,
            "eos_token_id": 1,
            "bos_token_id": 0,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 32,
            "patch_size": 16,
        },
        projection_dim=8,
    )
    transformers.CLIPModel(config).save_pretrained(d)
    return d


def test_clip_encoder_embeds_images_and_texts(tiny_clip_dir):
    from PIL import Image

    enc = im.ClipEncoder(str(tiny_clip_dir))
    red, blue = Image.new("RGB", (40, 40), (255, 0, 0)), Image.new("RGB", (40, 40), (0, 0, 255))
    v = im.embed_image(enc, red)
    assert v.shape == (8,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(v, im.embed_image(enc, blue))
    assert np.allclose(v, im.embed_image(enc, red), atol=1e-6)  # deterministic
    t = enc.texts(["a cozy room", "a big pool"])
    assert t.shape == (2, 8)