"""Phase 5: Image features from CLIP ViT-B/32 (rins on a normal CPU).

The dataset already contains a 512-number CLIP vector for each listing's main photo,
made with ``openai/clip-vit-base-patch32``. This module:

        verify  Loads the same CLIP model, embeds ~40 real listing photos and checks they
                match the dataset's vectors (so the web app can embed uploads the same way).
                Also embeds the zero-shot text prompts from config.yaml.
        train   Image-only price model (ridge regression on the 512-d vector) on the 
                Phase 1 folds, plus readable "prompt scores" such as how much a photo looks
                like "a luxurious interior" or "a small, cramped room".
                
Usage:
    python -m src.image_model verify    # downloads CLIP (~600 MB) once, ~2 minutes
    python -m src.image_model train     # ~1 minute
    
Listings without a photo vector keep ``has_image = 0`` and empty image columns; the 
fusion model (Phase 6) learns to ignore them.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV

from src.data import md
from src.metrics import metrics_by, price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)


## ---------------------------------------------------------------------------- CLIP
class Encoder(Protocol):
    """Anything that turns images and texts into vectors of the same size."""

    def images(self, images: list[Any]) -> np.ndarray: ...
    def texts(self, texts: list[str]) -> np.ndarray: ...


class ClipEncoder:
    """CLIP image and text encoder from Hugging Face ``transformers`` (CPU is fine)."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def images(self, images: list[Any]) -> np.ndarray:
        """PIL Images -> (n, 512) vectors, before normalization."""
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
            out = self.model.visual_projection(vision.pooler_output)
        return out.float().numpy()

    def texts(self, texts: list[str]) -> np.ndarray:
        """Strings -> (n, 512) vectors, before normalization."""
        with self.torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            text = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            out = self.model.text_projection(text.pooler_output)
        return out.float().numpy()


def unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (zero rows stay zero). CLIP compares by angle."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def embed_image(encoder: Encoder, image: Any) -> np.ndarray:
    """One PIL Image -> one unit-length 512-d vector."""
    return unit(encoder.images([image]))[0]


## ------------------------------------------------------------------------ FEATURES
def image_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Unit-length image vectors for every row, and a 0/1 mask of rows that have one."""
    has = df["has_image_emb"].to_numpy().astype(bool)
    dim = int(df.filter(pl.col("has_image_emb") == 1)["image_embeddings"].list.len().max())
    x = np.zeros((df.height, dim), dtype=np.float32)
    rows = np.where(has)[0]
    x[rows] = unit(np.array(df["image_embeddings"].gather(rows).to_list(), dtype=np.float32))
    return x, has

def prompt_scores(x: np.ndarray, prompt_vecs: np.ndarray, names: list[str]) -> np.ndarray:
    """How much each photo matches each prompt, relative to that photo's average match.
    
    Raw CLIP similaritites differ a lot from photo to photo; substracting the photo's mean
    over all prompts leaves "more like X than like the other prompts".
    """
    sims = unit(x) @ unit(prompt_vecs).T
    return sims - sims.mean(axis=1, keepdims=True)


def load_prompts(cfg: dict) -> tuple[list[str], np.ndarray] | None:
    """Prompt names and vectors saved by ``verify``; None if verify has not run yet."""
    path = Path(cfg["paths"]["models"]) / "clip_prompts.json"
    if not path.exists():
        return None

    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["prompts"] != cfg["image_model"]["prompts"]:
        log.warning("Prompts in config.yaml changed since `verify`; run verify again.")
        return None
    return list(saved["prompts"]), np.array(saved["vectors"], dtype=np.float32)


## ------------------------------------------------------------------------------- VERIFY

def download_photo(url: str, timeout: float) -> Any | None:
    """Fetch one photo as an RGB PIL image; None on any failure."""
    import requests
    from PIL import Image


    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def verify(cfg: dict, encoder: Encoder | None = None, fetch=download_photo) -> dict[str, Any]:
    """Re-embed real photos, compare with the dataset's vectors, save prompt vectors."""
    m = cfg["image_model"]
    encoder = encoder or ClipEncoder(m["clip_model"])
    df = pl.read_parquet(cfg["paths"]["processed"]).filter(pl.col("has_image_emb") == 1)
    df = df.sample(fraction=1.0, shuffle=True, seed=cfg["seed"])
    cosines, tried = [], 0
    for row in df.select("picture_url", "image_embeddings").iter_rows(named=True):
        if len(cosines) >= m["verify_photos"] or tried >= 4 * m["verify_photos"]:
            break
        tried += 1
        img = (
            fetch(row["picture_url"], cfg["checks"]["url_timeout_s"])
            if row["picture_url"]
            else None
        )
        if img is None:
            continue
        mine = embed_image(encoder, img)
        theirs = unit(np.array(row["image_embeddings"], dtype=np.float32))
        cosines.append(float(mine @ theirs))
    if not cosines:
        raise RuntimeError("No photo could be downloaded; check your internet connection.")

    names = list(m["prompts"])
    vectors = unit(encoder.texts([m["prompts"][n] for n in names]))
    out = Path(cfg["paths"]["models"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "clip_prompts.json").write_text(
        json.dumps(
            {"model": m["clip_model"], "prompts": m["prompts"], "vectors": vectors.tolist()}
        ),
        encoding="utf-8"
    )
    c = np.array(cosines)
    result = {
        "clip_model": m["clip_model"],
        "photos_tried": tried,
        "photos_compared": len(c),
        "cosine_median": round(float(np.median(c)), 4),
        "cosine_min": round(float(c.min()), 4),
        "matches": bool(np.median(c) >= m["verify_min_cosine"]),
    }
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "image_verify.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    verdict = "MATCH" if result["matches"] else "NO MATCH"
    print(
        f"Compared {len(c)} photos: median cosine {result['cosine_median']} "
        f"(min {result['cosine_min']}) -> {verdict}"
    )
    return result


## -------------------------------------------------------------------------------------- TRAIN
def cross_validate(df: pl.DataFrame, x: np.ndarray, has: np.ndarray, cfg: dict) -> np.ndarray:
    """Out-of-fold log-price predictions from the image alone (NaN where no image)."""
    alphas = [float(a) for a in cfg["image_model"]["probe_alphas"]]
    y = df["log_price"].to_numpy()
    folds = df["fold"].to_numpy()
    pred = np.full(df.height, np.nan)
    for k in np.unique(folds):
        tr = (folds != k) & has
        va = (folds == k) & has
        model = RidgeCV(alphas=alphas).fit(x[tr], y[tr])
        pred[va] = model.predict(x[va])
        log.info("fold %d: alpha %.3g", k, model.alpha_)
    return pred


def prompt_correlations(oof: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Spearman between each prompt score and the within-market log price."""
    d = oof.filter(pl.col("has_image") == 1).with_columns(
        (pl.col("log_price") - pl.col("log_price").median().over("market")).alias("rel")
    )
    rows = [
        {
            "prompt": n,
            "spearman_vs_price": round(
                float(d.select(pl.corr(f"img_{n}", "rel", method="spearman")).item()), 3
            ),
        }
        for n in names
    ]
    return pl.DataFrame(rows).sort(pl.col("spearman_vs_price").abs(), descending=True)


def write_report(path: Path, s: dict, t: dict[str, pl.DataFrame], prompts: dict[str, str]) -> None:
    v = s.get("verify")
    lines = [
        "# Image model: CLIP ViT-B/32",
        "_Generated by `python -m src.image_model`. Do not edit by hand._",
        f"\nPhoto vectors: `{s['clip_model']}` (512 numbers per listing, from the dataset). "
        f"{s['with_image']} of {s['rows']} listings ({s['with_image_pct']}%) have one.",
        "\n## 1. Verification",
        (
            f"Re-embedded {v['photos_compared']} real photos with the same model: median cosine "
            f"**{v['cosine_median']}** (min {v['cosine_min']}) with the dataset's vectors -> "
            f"**{'match' if v['matches'] else 'NO MATCH'}**. Uploaded photos in the web app "
            "can be embedded the same way."
            if v
            else "Not run yet: `python -m src.image_model verify`." 
        ),
        "\n## 2. Image-only model (out-of-fold, listings with a photo, USD)",
        "Ridge regression on the photo vector, strength chosen inside each training fold. "
        "The baseline row is the Phase 3 LightGBM on the **same** listings.",
        md(t["overall"]),
        "\n## 3. Per market (image-only)",
        md(t["by_market"]),
    ]
    if "prompts" in t:
        lines += [
            "\n## 4. Readable photo features: zero-shot prompt scores",
            "How much a photo looks like each phrase compared with the other phrases, "
            "correlated with the **within-market** price (positive = pricier than similar "
            "listings in the same city). These become explainable features in Phases 6 and 8.",
            md(
                t["prompts"]
                .join(
                    pl.DataFrame({"prompt": list(prompts), "text": list(prompts.values())}),
                    on="prompt",
                )
                .sort(pl.col("spearman_vs_price").abs(), descending=True)
            ),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Image-only cross-validation, prompt features, outputs and MLFlow logging."""
    set_seed(cfg["seed"])
    m, paths = cfg["image_model"], cfg["paths"]
    reports, preds = Path(paths["reports"]), Path(paths["predictions"])
    for p in (reports, preds):
        p.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(paths["processed"]).sort("listing_id")
    x, has = image_matrix(df)
    pred_log = cross_validate(df, x, has, cfg)

    oof = df.select("listing_id", "fold", "market", "room_type", "price", "log_price").with_columns(
        pl.Series("has_image", has.astype(np.int8)),
        pl.Series("pred_log", pred_log),
        pl.Series("pred", np.expm1(pred_log)),
    )
    prompts = load_prompts(cfg)
    if prompts:
        names, vecs = prompts
        scores = prompt_scores(x, vecs, names)
        scores[~has] = np.nan  
        oof = oof.with_columns(pl.Series(f"img_{n}", scores[:, i]) for i, n in enumerate(names))
    else:
        log.warning("No prompt vectors yet: run `python -m src.image_model verify` first.")
    oof.write_parquet(preds / "image_oof.parquet")

    sub = oof.filter(pl.col("has_image") == 1)
    rows = [
        {
            "model": "Image only (CLIP + ridge)",
            **price_metrics(sub["price"].to_numpy(), sub["pred"].to_numpy()),
        }
    ]
    base_path = preds / "baseline_oof.parquet"
    if base_path.exists():
        base = sub.select("listing_id", "price").join(
            pl.read_parquet(base_path).select("listing_id", "pred"), on="listing_id"
        )
        rows.insert(
            0,
            {
                "model": "LightGBM baseline (same listings)",
                **price_metrics(base["price"].to_numpy(), base["pred"].to_numpy()),
            },
        )
    verify_path = reports / "image_verify.json"
    summary = {
        "clip_model": m['clip_model'],
        "rows": df.height,
        "with_image": int(has.sum()),
        "with_image_pct": round(100 * float(has.mean()), 1),
        "overall": rows,
        "verify": json.loads(verify_path.read_text(encoding="utf-8"))
        if verify_path.exists()
        else None,
    }
    tables = {"overall": pl.DataFrame(rows), "by_market": metrics_by(sub, "market")}
    if prompts:
        tables['prompts'] = prompt_correlations(oof, prompts[0])
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
        write_report(reports / "image_model.md", summary, tables, m["prompts"])
    (reports / "image_model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with mlflow_run(cfg, "image_clip") as mlflow:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "clip_model": m["clip_model"],
                    "with_image": summary["with_image"],
                    "n_prompts": len(m["prompts"]) if prompts else 0,
                }
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in rows[-1].items() if k != "model"})
            for p in (reports / "image_model.md", reports / "image_model_metrics.json"):
                mlflow.log_artifact(str(p))
    r = rows[-1]
    print(
        f"Image only: MdAPE {r['mdape_pct']}%, within 10% {r['within_10_pct']}% "
        f"on {summary['with_image']} listings. Report: {reports / 'image_model.md'}"
    )
    return summary
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 image model")
    parser.add_argument("step", choices=["verify", "train"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    verify(cfg) if args.step == "verify" else run(cfg)
 
 
if __name__ == "__main__":
    main()