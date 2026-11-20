"""Phase 9 tests: the price service and monitoring, on synthetic data with every model saved."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from src import baseline as bl  # noqa: E402
from src import fusion as fu  # noqa: E402
from src import intervals as iv  # noqa: E402
from src import monitor as mon  # noqa: E402
from src import serve as sv  # noqa: E402
from src.utils import load_config  # noqa: E402
from tests.test_fusion import write_inputs  # noqa: E402


@pytest.fixture(scope="module")
def cfg(tmp_path_factory) -> dict:
    """All Phase 1-8 outputs, built once for this file (about 20 seconds)."""
    tmp = tmp_path_factory.mktemp("serve")
    c = copy.deepcopy(load_config("config.yaml"))
    c["paths"] = {
        "raw": str(tmp / "raw.parquet"),
        "processed": str(tmp / "listings.parquet"),
        "reports": str(tmp / "reports"),
        "predictions": str(tmp / "predictions"),
        "models": str(tmp / "models"),
    }
    c["mlflow"]["enabled"] = False
    c["fusion"].update(seeds=1, batch_size=64)
    c["intervals"]["difficulty"]["num_boost_round"] = 50
    c["serve"].update(log_dir=str(tmp / "monitoring"), use_photo=False, min_market_rows=20)
    df = write_inputs(c, 500)
    models, preds = Path(c["paths"]["models"]), Path(c["paths"]["predictions"])
    booster, fb = bl.train_final(df, c, 100)
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / "baseline_lightgbm.txt"))
    (models / "baseline_features.json").write_text(json.dumps(fb.to_dict()), encoding="utf-8")
    fu.train_final(fu.load_inputs(c), c, "concat", list(fu.MODALITIES), 5, models / "fusion_concat")
    pl.read_parquet(preds / "baseline_oof.parquet").write_parquet(preds / "best_oof.parquet")
    (preds / "best_model.json").write_text(json.dumps({"best": "Synthetic"}), encoding="utf-8")
    iv.run(c)
    rng = np.random.default_rng(0)  # 3 photo prompts, as in tests.test_fusion
    prompts = {
        "prompts": {"luxury": "", "cramped": "", "pool": ""},
        "vectors": rng.normal(size=(3, 512)).tolist(),
    }
    (models / "clip_prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    (models / "serving").mkdir(exist_ok=True)
    (models / "serving" / "form.json").write_text(json.dumps(sv.build_form(c)), encoding="utf-8")
    mon.build_reference(c)
    return c


@pytest.fixture(scope="module")
def predictor(cfg) -> sv.Predictor:
    return sv.Predictor(cfg)


def a_listing(predictor: sv.Predictor, **kw) -> sv.Listing:
    market = next(iter(predictor.form["markets"]))
    return sv.Listing(market=market, **kw)


# ---------------------------------------------------------------- building the form
def test_form_has_markets_typical_values_and_locations(predictor):
    form = predictor.form
    assert form["markets"] and form["room_types"] and "accommodates" in form["numbers"]
    m = next(iter(form["markets"].values()))
    assert m["typical"]["accommodates"] >= 1
    hood = next(iter(m["neighbourhoods"].values()))
    assert {"latitude", "longitude"} <= set(hood)


def test_listing_row_applies_the_users_choices(predictor):
    x = a_listing(
        predictor, accommodates=6, amenities=["amen_pool"], title="Sea view", description="Big"
    )
    row = predictor.listing_row(x).row(0, named=True)
    assert row["accommodates"] == 6 and row["amen_pool"] == 1 and row["amen_wifi"] == 0
    assert row["text"] == "Sea view . Big"
    with pytest.raises(ValueError, match="Unknown market"):
        predictor.listing_row(sv.Listing(market="Atlantis"))


# ---------------------------------------------------------------- predictions
def test_prediction_has_price_range_review_flag_and_reasons(predictor):
    out = predictor.predict(a_listing(predictor))
    assert out["model"] == "fusion" and out["price"] > 0
    assert out["range"]["low"] <= out["price"] <= out["range"]["high"]
    assert isinstance(out["review"], bool) and len(out["reasons"]) == 5
    assert not out["used_photo"] and not out["used_text"]  # no photo, no text model


def test_more_guests_means_a_higher_price(predictor):
    small = predictor.predict(a_listing(predictor, accommodates=1, bedrooms=0))
    big = predictor.predict(a_listing(predictor, accommodates=9, bedrooms=4))
    assert big["price"] > small["price"]


def test_photo_vector_is_used_by_the_fusion_model(predictor):
    vec = np.random.default_rng(1).normal(size=512).astype(np.float32)
    out = predictor.predict(a_listing(predictor), photo_vec=vec / np.linalg.norm(vec))
    assert out["used_photo"] and np.isfinite(out["price"])


def test_baseline_mode_serves_lightgbm_only(cfg):
    c = copy.deepcopy(cfg)
    c["serve"]["model"] = "baseline"
    p = sv.Predictor(c)
    out = p.predict(a_listing(p))
    assert out["model"] == "baseline" and out["price"] == out["baseline_price"]


# ---------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(cfg, predictor) -> TestClient:
    return TestClient(sv.create_app(cfg, predictor))


def test_page_form_and_health(client):
    assert "Listing price estimator" in client.get("/").text
    assert client.get("/form").json()["markets"]
    assert client.get("/health").json()["status"] == "ok"


def test_predict_endpoint_logs_and_accepts_feedback(client, cfg, predictor):
    body = {"listing": a_listing(predictor, accommodates=3).model_dump_json()}
    r = client.post("/predict", data=body)
    assert r.status_code == 200, r.text
    out = r.json()
    logged = Path(cfg["serve"]["log_dir"]) / "requests.jsonl"
    assert out["id"] in logged.read_text(encoding="utf-8")
    assert (
        client.post("/feedback", json={"id": out["id"], "actual_price": 120.0}).status_code == 200
    )


def test_predict_rejects_bad_input(client, predictor):
    bad = a_listing(predictor).model_dump() | {"accommodates": 0}
    assert client.post("/predict", data={"listing": json.dumps(bad)}).status_code == 422
    assert client.post("/predict", data={"listing": '{"market": "Atlantis"}'}).status_code == 422
    photo = {"photo": ("x.jpg", b"not an image", "image/jpeg")}
    r = client.post(
        "/predict", data={"listing": a_listing(predictor).model_dump_json()}, files=photo
    )
    assert r.status_code == 422


# ---------------------------------------------------------------- monitoring
def test_psi_known_values():
    assert mon.psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert mon.psi([0.5, 0.5], [0.9, 0.1]) > 0.2
    s = mon.shares(np.array([1.0, 2.0, np.nan, 5.0]), [1.5, 3.0])
    assert s == [0.25, 0.25, 0.25, 0.25]  # 3 bins + missing


def test_simulation_reports_outcomes_and_market_drift(cfg):
    c = copy.deepcopy(cfg)
    c["serve"]["log_dir"] = str(Path(cfg["serve"]["log_dir"]).parent / "sim")
    c["monitor"]["min_feedback"] = 10
    mon.simulate(c, 80, None, 0)
    s = mon.summarise(c)
    assert s["requests"] == 80 and s["outcomes"]["with_actual_price"] == 80
    assert 0 <= s["outcomes"]["range_coverage_pct"] <= 100
    market_psi = {d["feature"]: d["psi"] for d in s["drift"]}["market"]

    one = sorted(cfg_markets := list(sv.Predictor(c).form["markets"]))[0]
    mon.simulate(c, 200, one, 1)  # a week of only one market
    s2 = mon.summarise(c)
    shifted = {d["feature"]: d for d in s2["drift"]}["market"]
    assert shifted["psi"] > market_psi and shifted["status"] != "stable", cfg_markets
    assert any("market" in a for a in s2["alerts"])
    mon.write_report(Path(c["paths"]["reports"]) / "monitoring.md", s2)
    assert "## 2. Drift" in (Path(c["paths"]["reports"]) / "monitoring.md").read_text(
        encoding="utf-8"
    )


def test_description_is_embedded_with_the_text_model(cfg, tmp_path):
    """A tiny DistilBERT (16-d, like the synthetic text vectors) saved as Phase 4 would."""
    import shutil

    import transformers

    from src import text_model as tm
    from tests.test_text_model import WORDS

    d = tmp_path / "tiny"
    d.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    config = transformers.DistilBertConfig(
        vocab_size=len(vocab),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=64,
    )
    transformers.DistilBertModel(config).save_pretrained(d)

    c = copy.deepcopy(cfg)
    c["paths"]["models"] = str(tmp_path / "models")
    shutil.copytree(cfg["paths"]["models"], c["paths"]["models"])
    c["text_model"].update(model_name=str(d), max_length=16, batch_size=16, warmup_frac=0.0)
    df = pl.read_parquet(c["paths"]["processed"])
    tm.train_final(df, c, 1, Path(c["paths"]["models"]) / "text_model")

    p = sv.Predictor(c)
    with_text = p.predict(a_listing(p, description=" ".join(WORDS[:6])))
    without = p.predict(a_listing(p))
    assert with_text["used_text"] and not without["used_text"]
    assert np.isfinite(with_text["price"])


def test_uploaded_photo_goes_through_clip(cfg):
    """A fake CLIP (no download): the upload path embeds the photo and the model uses it."""
    import io

    from PIL import Image

    class FakeClip:
        def images(self, images):
            return np.random.default_rng(0).normal(size=(len(images), 512)).astype(np.float32)

    c = copy.deepcopy(cfg)
    c["serve"]["use_photo"] = True
    p = sv.Predictor(c)
    p._clip = FakeClip()
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, "JPEG")
    client = TestClient(sv.create_app(c, p))
    files = {"photo": ("room.jpg", buf.getvalue(), "image/jpeg")}
    r = client.post("/predict", data={"listing": a_listing(p).model_dump_json()}, files=files)
    assert r.status_code == 200 and r.json()["used_photo"] and r.json()["note"] is None

    def broken(_image):
        raise OSError("CLIP could not be downloaded")

    p.photo_vector = broken  # the price still comes back, without the photo
    r = client.post("/predict", data={"listing": a_listing(p).model_dump_json()}, files=files)
    assert r.status_code == 200 and not r.json()["used_photo"] and "photo" in r.json()["note"]
