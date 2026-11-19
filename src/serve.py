"""Phase 9: price API and web page. Runs on a CPU.

Usage:
    python -m src.serve build      # once: form options and typical values per market
    python -m src.serve run        # open http://127.0.0.1:8000

One request = one NEW listing described on the web page. Fields that are not on the form
(review scores, host details, ...) are filled with the typical value for that market.

Pipeline for one request:
1. LightGBM baseline (Phase 3) on the table features  -> base price
2. Fusion network (Phase 6) adds a correction from the description (Phase 4 text model)
   and the photo (CLIP, Phase 5). Missing text or photo is masked, as in training.
3. Calibrator (Phase 8) -> 80% price range and the human-review decision
4. SHAP (Phase 8) -> the top reasons
Every request is appended to data/monitoring/requests.jsonl for Phase 9 monitoring.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from src.baseline import CATEGORICAL
from src.explain import explain_listing, load_baseline
from src.intervals import Calibrator
from src.utils import load_config, setup_logging

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / "web"
FORM_NUMBERS = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights"]
NOT_FEATURES = {"listing_id", "price", "log_price", "price_local", "fold"}


# --------------------------------------------------------------------------- build
def build_form(cfg: dict[str, Any], df: pl.DataFrame | None = None) -> dict[str, Any]:
    """Dropdown options and a 'typical listing' per market, from the processed data."""
    df = df if df is not None else pl.read_parquet(cfg["paths"]["processed"])
    df = df.with_columns(pl.col("market").fill_null(""))
    s = cfg["serve"]
    numeric = [
        c
        for c, t in df.schema.items()
        if t.is_numeric() and c not in NOT_FEATURES and not c.startswith("has_")
    ]
    cats = [c for c in CATEGORICAL if c in df.columns]
    markets = {}
    counts = df["market"].value_counts().sort("count", descending=True)
    for m, n in counts.iter_rows():
        if n < s["min_market_rows"] or not m:
            continue
        part = df.filter(pl.col("market") == m)
        typical = {c: part[c].median() for c in numeric}
        typical |= {
            c: part[c].drop_nulls().mode().sort()[0] for c in cats if part[c].null_count() < n
        }
        hoods = (
            part.filter(pl.col("neighbourhood").is_not_null())
            .group_by("neighbourhood")
            .agg(
                pl.len().alias("n"),
                *[
                    pl.col(c).median()
                    for c in ("latitude", "longitude", "dist_centre_km")
                    if c in df.columns
                ],
            )
            .filter(pl.col("n") >= 3)
            .sort("neighbourhood")
        )
        markets[m] = {
            "rows": int(n),
            "typical": {
                k: (float(v) if isinstance(v, int | float) else v) for k, v in typical.items()
            },
            "neighbourhoods": {r.pop("neighbourhood"): r for r in hoods.drop("n").to_dicts()},
        }
    return {
        "markets": markets,
        "room_types": sorted(df["room_type"].drop_nulls().unique().to_list()),
        "property_types": df["property_type"]
        .value_counts()
        .sort("count", descending=True)["property_type"]
        .drop_nulls()
        .head(15)
        .to_list(),
        "numbers": [c for c in FORM_NUMBERS if c in df.columns],
        "amenities": sorted(c for c in df.columns if c.startswith("amen_")),
    }


# --------------------------------------------------------------------------- request
class Listing(BaseModel):
    """What the web page sends. Anything left out takes the market's typical value."""

    market: str
    neighbourhood: str | None = None
    room_type: str | None = None
    property_type: str | None = None
    accommodates: float | None = Field(None, ge=1, le=30)
    bedrooms: float | None = Field(None, ge=0, le=20)
    beds: float | None = Field(None, ge=0, le=30)
    bathrooms: float | None = Field(None, ge=0, le=20)
    minimum_nights: float | None = Field(None, ge=1, le=365)
    amenities: list[str] = []
    title: str = ""
    description: str = ""


class Feedback(BaseModel):
    id: str
    actual_price: float = Field(gt=0)


def listing_row(form: dict[str, Any], x: Listing, known: dict[str, Any] | None = None) -> dict:
    """The request as one row of the processed table (as a dict)."""
    if x.market not in form["markets"]:
        raise ValueError(f"Unknown market {x.market!r}")
    mk = form["markets"][x.market]
    row = dict(mk["typical"]) | {"listing_id": "request", "market": x.market}
    if known:  # monitoring replays: the listing's real values for fields not on the form
        row |= {k: v for k, v in known.items() if k in row and v is not None}
    if x.neighbourhood:
        row["neighbourhood"] = x.neighbourhood
        row |= mk["neighbourhoods"].get(x.neighbourhood, {})
    for c in ("room_type", "property_type", *form["numbers"]):
        if getattr(x, c, None) is not None:
            row[c] = getattr(x, c)
    if form["amenities"]:
        chosen = set(x.amenities) & set(form["amenities"])
        before = sum(row.get(a, 0) or 0 for a in form["amenities"])
        for a in form["amenities"]:
            row[a] = 1 if a in chosen else 0
        if "amenities_count" in row:
            row["amenities_count"] = max(row["amenities_count"] - before + len(chosen), len(chosen))
    text = " . ".join(t.strip() for t in (x.title, x.description) if t.strip())
    return row | {"text": text, "text_n_words": len(text.split())}


def listing_from_row(r: dict[str, Any], form: dict[str, Any]) -> Listing:
    """What a host would type into the form for an existing listing (used by replays)."""
    fields = set(Listing.model_fields) - {"amenities", "description"}
    values = {k: v for k, v in r.items() if k in fields and v is not None}
    for k, v in values.items():  # real data can exceed the form's limits (e.g. 1,250 nights)
        for limit in Listing.model_fields[k].metadata:
            v = max(v, limit.ge) if hasattr(limit, "ge") else v
            v = min(v, limit.le) if hasattr(limit, "le") else v
        values[k] = v
    return Listing(
        **values,
        amenities=[a for a in form["amenities"] if r.get(a) == 1],
        description=r.get("text") or "",
    )


class Predictor:
    """Loads every trained part once; ``predict`` prices one listing."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg, s = cfg, cfg["serve"]
        models = Path(cfg["paths"]["models"])
        form_path = models / "serving" / "form.json"
        if not form_path.exists():
            raise FileNotFoundError(f"{form_path} not found: run `python -m src.serve build` first")
        self.form = json.loads(form_path.read_text(encoding="utf-8"))
        self.booster, self.fb = load_baseline(models)
        self.calibrator = Calibrator.load(models / "intervals")
        form_cal = models / "intervals_form"  # Phase 10: ranges sized for form-only input
        self.form_calibrator = Calibrator.load(form_cal) if form_cal.exists() else None
        self.fusion = self._load_fusion(models) if s["model"] == "fusion" else None
        self.model_name = "fusion" if self.fusion else "baseline"
        self.text = self._load_text(models) if self.fusion and s["use_text_model"] else None
        self.prompts = self._load_prompts(models)
        self._clip: Any = None  # loaded on the first uploaded photo
        log.info("Serving the %s model (text model: %s)", self.model_name, self.text is not None)

    # ---- optional parts
    def _load_fusion(self, models: Path) -> dict[str, Any] | None:
        d = models / "fusion_concat"
        if not (d / "model.pt").exists():
            log.warning("No fusion model in %s: serving the LightGBM baseline", d)
            return None
        import torch

        from src import fusion as fu

        meta = json.loads((d / "fusion.json").read_text(encoding="utf-8"))
        tab = fu.TabPreprocessor.from_dict(meta["tab"])
        f = meta["fusion"]
        net = fu.FusionNet(
            meta["arch"],
            meta["modalities"],
            tab.n_numeric,
            tab.cat_sizes,
            meta["d_text"],
            meta["d_img"],
            f["hidden"],
            f["heads"],
            f["dropout"],
            f["residual_on_baseline"],
        )
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        return {"net": net.eval(), "tab": tab, "meta": meta}

    def _load_text(self, models: Path) -> tuple[Any, Any, dict] | None:
        d = models / "text_model"
        if not (d / "text_model.json").exists():
            log.warning("No text model in %s: descriptions will be ignored", d)
            return None
        from src.text_model import load_final

        return load_final(d)

    def _load_prompts(self, models: Path) -> np.ndarray | None:
        path = models / "clip_prompts.json"
        if not path.exists():
            return None
        return np.array(json.loads(path.read_text(encoding="utf-8"))["vectors"], dtype=np.float32)

    def clip(self) -> Any:
        if self._clip is None:
            from src.image_model import ClipEncoder

            self._clip = ClipEncoder(self.cfg["image_model"]["clip_model"])
        return self._clip

    # ---- one listing
    def listing_row(self, x: Listing, known: dict[str, Any] | None = None) -> pl.DataFrame:
        return pl.DataFrame([listing_row(self.form, x, known)], infer_schema_length=None)

    def photo_vector(self, image: Any) -> np.ndarray:
        from src.image_model import embed_image

        return embed_image(self.clip(), image.convert("RGB"))

    def fusion_log_price(
        self, row: pl.DataFrame, base_log: float, text: str, vec: np.ndarray | None
    ) -> float:
        import torch

        from src import fusion as fu
        from src.image_model import prompt_scores
        from src.text_model import tokenize

        f, meta = self.fusion, self.fusion["meta"]
        mu, sd = meta["target_mean"], meta["target_std"]
        num, cat = f["tab"].transform(row)
        n_prompts = len(meta["prompt_mean"])
        img = np.zeros((1, meta["d_img"]), dtype=np.float32)
        has = vec is not None and (n_prompts == 0 or self.prompts is not None)
        if has:
            img[0, : vec.shape[0]] = vec
            if n_prompts:
                p = prompt_scores(vec[None], self.prompts, [""] * len(self.prompts))[0]
                img[0, vec.shape[0] :] = (p - np.array(meta["prompt_mean"])) / np.array(
                    meta["prompt_std"]
                )
        t = np.zeros((1, meta["d_text"]), dtype=np.float32)
        use_text = self.text is not None and bool(text.strip())
        if use_text:
            model, tokenizer, tmeta = self.text
            ids = tokenize(tokenizer, [text], tmeta["max_length"])[0]
            with torch.no_grad():
                t = model.embed(
                    torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long)
                ).numpy()
        tt = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)  # noqa: E731
        a = fu.Arrays(
            tt(num),
            tt(cat, torch.long),
            tt(t),
            tt(img),
            tt([float(has)]),
            tt([(base_log - mu) / sd]),
            tt([0.0]),
        )
        with torch.no_grad():
            out, _ = f["net"](a, drop_text=torch.tensor([not use_text]))
        return float(out[0]) * sd + mu

    def predict(
        self,
        x: Listing,
        image: Any = None,
        photo_vec: np.ndarray | None = None,
        known: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        row = self.listing_row(x, known)
        note = None
        if image is not None and self.cfg["serve"]["use_photo"]:
            try:
                photo_vec = self.photo_vector(image)
            except Exception:  # e.g. CLIP cannot be downloaded: price without the photo
                log.exception("Photo could not be processed")
                note = "The photo could not be processed, so it was not used."
        base_log = float(self.booster.predict(self.fb.matrix(row))[0])
        pred_log = base_log
        if self.fusion:
            pred_log = self.fusion_log_price(row, base_log, row["text"][0], photo_vec)
        price = float(np.expm1(pred_log))
        # A web request has only the form fields: use the ranges sized for that (Phase 10).
        new_listing = known is None and self.form_calibrator is not None
        cal = self.form_calibrator if new_listing else self.calibrator
        r = cal.apply(row.with_columns(pl.lit(pred_log).alias("pred_log"))).row(0, named=True)
        k = f"{round(cal.levels[0] * 100)}"
        why = explain_listing(self.booster, self.fb, row, top=5)
        return {
            "id": uuid.uuid4().hex[:12],
            "model": self.model_name,
            "price": round(price, 2),
            "baseline_price": round(float(np.expm1(base_log)), 2),
            "range": {
                "level": int(k),
                "low": round(r[f"lo_{k}"], 2),
                "high": round(r[f"hi_{k}"], 2),
            },
            "range_basis": "form only (new listing)" if new_listing else "full listing data",
            "review": bool(r["review"]),
            "review_reason": r["reason"] or None,
            "reasons": why["reasons"],
            "typical_price": why["typical_price"],
            "used_photo": photo_vec is not None and self.fusion is not None,
            "used_text": self.text is not None and bool(row["text"][0].strip()),
            "note": note,
            "latency_ms": round(1000 * (time.perf_counter() - start), 1),
        }


# --------------------------------------------------------------------------- logging
def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_request(cfg: dict[str, Any], x: Listing, out: dict[str, Any]) -> None:
    append(
        Path(cfg["serve"]["log_dir"]) / "requests.jsonl",
        {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            **x.model_dump(exclude={"title", "description", "amenities"}),
            "n_amenities": len(x.amenities),
            **{
                k: out[k]
                for k in (
                    "id",
                    "model",
                    "price",
                    "review",
                    "review_reason",
                    "used_photo",
                    "used_text",
                    "latency_ms",
                )
            },
            "low": out["range"]["low"],
            "high": out["range"]["high"],
        },
    )


# --------------------------------------------------------------------------- API
def create_app(cfg: dict[str, Any] | None = None, predictor: Predictor | None = None) -> FastAPI:
    cfg = cfg or load_config()
    p = predictor or Predictor(cfg)
    logs = Path(cfg["serve"]["log_dir"])
    app = FastAPI(title="Airbnb price estimator")

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/form")
    def form() -> dict[str, Any]:
        return {**p.form, "model": p.model_name, "text_model": p.text is not None}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": p.model_name,
            "text_model": p.text is not None,
            "photo": p.cfg["serve"]["use_photo"],
        }

    @app.post("/predict")
    async def predict(
        listing: Annotated[str, Form()], photo: Annotated[UploadFile | None, File()] = None
    ) -> dict[str, Any]:
        try:
            x = Listing.model_validate_json(listing)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e
        image = None
        if photo is not None and photo.filename:
            from PIL import Image

            try:
                image = Image.open(io.BytesIO(await photo.read()))
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, "The photo could not be read as an image") from e
        try:
            out = p.predict(x, image)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception:
            append(
                logs / "errors.jsonl",
                {"time": datetime.now(UTC).isoformat(timespec="seconds"), "market": x.market},
            )
            log.exception("Prediction failed")
            raise HTTPException(500, "Prediction failed") from None
        log_request(cfg, x, out)
        return out

    @app.post("/feedback")
    def feedback(fb: Feedback) -> dict[str, str]:
        """The real price, once known: lets monitoring measure accuracy and range coverage."""
        append(
            logs / "feedback.jsonl",
            {"time": datetime.now(UTC).isoformat(timespec="seconds"), **fb.model_dump()},
        )
        return {"status": "saved"}

    @app.get("/monitoring")
    def monitoring() -> dict[str, Any]:
        from src.monitor import summarise

        return summarise(cfg)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: price API and web page")
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.command == "build":
        form = build_form(cfg)
        out = Path(cfg["paths"]["models"]) / "serving" / "form.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(form, indent=2), encoding="utf-8")
        from src.monitor import build_reference

        build_reference(cfg)
        print(f"{len(form['markets'])} markets saved to {out}")
    else:
        import uvicorn

        uvicorn.run(create_app(cfg), host=cfg["serve"]["host"], port=cfg["serve"]["port"])


if __name__ == "__main__":
    main()
