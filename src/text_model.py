"""Phase 4: fine-tune a Transformer on the listing text to predict log(price).

Usage (on a GPU, e.g. Google Colab; see notebooks/phase4_text_colab.ipynb):
    python -m src.text_model train --no-mlflow

Then, on your own computer, record the Colab results in your local MLflow:
    python -m src.text_model log

Design:
  * Multilingual DistilBERT: most markets write descriptions in Turkish,
    Spanish, Portuguese, French or Chinese, not only English.
  * Same 5 folds as the baseline. For each fold the model is fine-tuned on the
    training rows; the best epoch is picked on a 10% slice of the TRAINING rows.
  * Saves, per listing, the out-of-fold prediction and the out-of-fold text
    vector (mean-pooled encoder output). Each vector comes from a model that
    never saw that listing, so Phase 6 can fuse them without leakage.
  * Optionally trains a final model on all rows for the web app (Phase 9).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import md, outlier_bounds
from src.metrics import metrics_by, price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- model
class TextRegressor(nn.Module):
    """Transformer encoder + mean pooling + a linear head that outputs one number."""

    def __init__(self, encoder: nn.Module, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean of the token vectors, ignoring padding."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vec = self.embed(input_ids, attention_mask)
        return self.head(vec).squeeze(-1), vec


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
def tokenize(tokenizer: Any, texts: list[str], max_length: int) -> list[list[int]]:
    """Token ids per text (no padding yet; batches are padded on the fly)."""
    texts = [t if t and t.strip() else "[empty]" for t in texts]
    return tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]


def batches(ids: list[list[int]], order: np.ndarray, size: int, pad_id: int):
    """Yield (row positions, input_ids, attention_mask) padded to the longest text in the batch."""
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        longest = max(len(ids[i]) for i in rows)
        input_ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for j, i in enumerate(rows):
            input_ids[j, : len(ids[i])] = torch.tensor(ids[i])
            mask[j, : len(ids[i])] = 1
        yield rows, input_ids, mask


# --------------------------------------------------------------------------- training
def predict(
    model: TextRegressor, ids: list[list[int]], rows: np.ndarray, cfg: dict, pad_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised predictions and text vectors for ``rows`` (in the order of ``rows``)."""
    t = cfg["text_model"]
    dev = device()
    model.to(dev)
    model.eval()
    order = rows[np.argsort([len(ids[i]) for i in rows])]  # similar lengths -> less padding
    preds, vecs = {}, {}
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for r, x, m in batches(ids, order, t["batch_size"] * 2, pad_id):
            p, v = model(x.to(dev), m.to(dev))
            for i, pi, vi in zip(r, p.float().cpu().numpy(), v.float().cpu().numpy(), strict=True):
                preds[i], vecs[i] = pi, vi
    return np.array([preds[i] for i in rows]), np.stack([vecs[i] for i in rows])


def fine_tune(
    model: TextRegressor,
    ids: list[list[int]],
    y: np.ndarray,
    train_rows: np.ndarray,
    check_rows: np.ndarray | None,
    epochs: int,
    cfg: dict,
    pad_id: int,
    seed: int,
) -> tuple[TextRegressor, int, list[float]]:
    """Train on ``train_rows``; if ``check_rows`` is given, keep the epoch with the lowest
    loss on them. Returns (model, best epoch, loss per epoch on check_rows)."""
    t = cfg["text_model"]
    dev = device()
    model.to(dev)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": t["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(groups, lr=float(t["learning_rate"]))
    steps = epochs * math.ceil(len(train_rows) / t["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(t["warmup_frac"] * steps), steps)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_state, history = math.inf, epochs, None, []
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        for rows, x, m in batches(ids, rng.permutation(train_rows), t["batch_size"], pad_id):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                pred, _ = model(x.to(dev), m.to(dev))
                loss = nn.functional.mse_loss(pred.float(), y_t[rows].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        if check_rows is not None:
            p, _ = predict(model, ids, check_rows, cfg, pad_id)
            check_loss = float(np.mean((p - y[check_rows]) ** 2))
            history.append(round(check_loss, 4))
            log.info("  epoch %d: inner-check MSE %.4f", epoch, check_loss)
            if check_loss < best_loss:
                best_loss, best_epoch = check_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def new_model(cfg: dict) -> tuple[TextRegressor, Any]:
    """Fresh pretrained encoder + new head, and its tokenizer."""
    t = cfg["text_model"]
    tokenizer = AutoTokenizer.from_pretrained(t["model_name"])
    encoder = AutoModel.from_pretrained(t["model_name"])
    return TextRegressor(encoder, t["dropout"]), tokenizer


def keep_rows(df: pl.DataFrame, rows: np.ndarray, cfg: dict) -> np.ndarray:
    """Fold-wise outlier rule (same as the baseline): drop extreme TRAINING rows."""
    o = cfg["outliers"]
    y = df["log_price"].to_numpy()[rows]
    lo, hi = outlier_bounds(y, o["lower_quantile"], o["upper_quantile"])
    return rows[(y >= lo) & (y <= hi)]


def cross_validate(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Out-of-fold predictions and text vectors for every listing."""
    t = cfg["text_model"]
    _, tokenizer = new_model(cfg)
    ids = tokenize(tokenizer, df["text"].to_list(), t["max_length"])
    pad_id = tokenizer.pad_token_id
    folds = df["fold"].to_numpy()
    y_log = df["log_price"].to_numpy()
    pred_log = np.zeros(df.height)
    vecs: np.ndarray | None = None
    fold_rows = []
    for k in sorted(np.unique(folds)):
        start = time.time()
        set_seed(cfg["seed"] + int(k))
        train = keep_rows(df, np.where(folds != k)[0], cfg)
        valid = np.where(folds == k)[0]
        mu, sd = float(y_log[train].mean()), float(y_log[train].std())
        y_std = (y_log - mu) / sd  # standardised with TRAINING statistics only
        rng = np.random.default_rng(cfg["seed"] + int(k))
        inner = rng.random(len(train)) < t["inner_valid_frac"]
        model, _ = new_model(cfg)
        model, best_epoch, history = fine_tune(
            model, ids, y_std, train[~inner], train[inner], t["epochs"], cfg, pad_id, cfg["seed"]
        )
        p, v = predict(model, ids, valid, cfg, pad_id)
        pred_log[valid] = p * sd + mu
        if vecs is None:
            vecs = np.zeros((df.height, v.shape[1]), dtype=np.float32)
        vecs[valid] = v
        m = price_metrics(np.expm1(y_log[valid]), np.expm1(pred_log[valid]))
        fold_rows.append(
            {
                "fold": int(k),
                "train_rows": int((~inner).sum()),
                "valid_rows": len(valid),
                "best_epoch": best_epoch,
                "inner_mse_by_epoch": str(history),
                "minutes": round((time.time() - start) / 60, 1),
                **m,
            }
        )
        log.info(
            "fold %d: MdAPE %.2f%%, within 10%% %.1f%% (%.1f min)",
            k,
            m["mdape_pct"],
            m["within_10_pct"],
            (time.time() - start) / 60,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof = df.select("listing_id", "fold", "market", "room_type", "price").with_columns(
        pl.Series("pred_log", pred_log),
        pl.Series("pred", np.expm1(pred_log)),
        pl.Series("text_vec", vecs).cast(pl.List(pl.Float32)),
    )
    return oof, {"folds": pl.DataFrame(fold_rows)}


def train_final(df: pl.DataFrame, cfg: dict, epochs: int, out_dir: Path) -> None:
    """Fine-tune on all rows and save encoder, tokenizer, head and target scaling."""
    t = cfg["text_model"]
    set_seed(cfg["seed"])
    model, tokenizer = new_model(cfg)
    ids = tokenize(tokenizer, df["text"].to_list(), t["max_length"])
    rows = keep_rows(df, np.arange(df.height), cfg)
    y_log = df["log_price"].to_numpy()
    mu, sd = float(y_log[rows].mean()), float(y_log[rows].std())
    model, _, _ = fine_tune(
        model, ids, (y_log - mu) / sd, rows, None, epochs, cfg, tokenizer.pad_token_id, cfg["seed"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.half().save_pretrained(out_dir)  # fp16 halves the download size
    tokenizer.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    meta = {
        "model_name": t["model_name"],
        "max_length": t["max_length"],
        "dropout": t["dropout"],
        "target_mean": mu,
        "target_std": sd,
        "epochs": epochs,
    }
    (out_dir / "text_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_final(model_dir: Path) -> tuple[TextRegressor, Any, dict]:
    """Load the saved final model (used by the web app in Phase 9)."""
    meta = json.loads((model_dir / "text_model.json").read_text(encoding="utf-8"))
    encoder = AutoModel.from_pretrained(model_dir).float()  # saved in fp16; run in fp32
    model = TextRegressor(encoder, meta["dropout"])
    model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
    return model.eval(), AutoTokenizer.from_pretrained(model_dir), meta


# --------------------------------------------------------------------------- report
def baseline_row(reports: Path) -> dict | None:
    """The LightGBM baseline's overall metrics, if Phase 3 has been run."""
    path = reports / "baseline_metrics.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["overall"]
    return next((r for r in rows if r["model"] == "LightGBM baseline"), None)


def write_report(path: Path, s: dict, t: dict[str, pl.DataFrame]) -> None:
    lines = [
        "# Text model: fine-tuned multilingual DistilBERT",
        "_Generated by `python -m src.text_model`. Do not edit by hand._",
        f"\nModel `{s['model_name']}`, max {s['max_length']} tokens, {s['epochs']} epochs max "
        f"(best epoch chosen on 10% of each training fold), {s['rows']} listings, "
        "Phase 1 folds. Input: name + summary + description with prices masked. "
        "No structured features: this measures what the **text alone** knows.",
        "\n## 1. Overall (out-of-fold, USD)",
        md(t["overall"]),
        "\nThe text model is not expected to beat the baseline on its own. Its job is to add "
        "information the table lacks; Phase 6 combines both and Phase 7 tests the gain.",
        "\n## 2. Per fold",
        md(t["folds"]),
        "\n## 3. Per market",
        md(t["by_market"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cross-validate, optionally train the final model, write outputs."""
    set_seed(cfg["seed"])
    t, paths = cfg["text_model"], cfg["paths"]
    reports, preds_dir = Path(paths["reports"]), Path(paths["predictions"])
    for p in (reports, preds_dir):
        p.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(paths["processed"]).sort("listing_id")
    log.info("Text model on %d rows, device=%s, model=%s", df.height, device(), t["model_name"])
    if device().type == "cpu" and df.height > 1000:
        log.warning("No GPU found: this will be very slow. Use the Colab notebook.")

    oof, details = cross_validate(df, cfg)
    oof.write_parquet(preds_dir / "text_oof.parquet")
    y, pred = oof["price"].to_numpy(), oof["pred"].to_numpy()
    rows = [{"model": "Text (DistilBERT)", **price_metrics(y, pred)}]
    base = baseline_row(reports)
    if base:
        rows.insert(0, base)
    overall = pl.DataFrame(rows)
    final_epochs = max(1, int(round(float(np.median(details["folds"]["best_epoch"])))))
    summary = {
        "model_name": t["model_name"],
        "max_length": t["max_length"],
        "epochs": t["epochs"],
        "rows": df.height,
        "final_epochs": final_epochs,
        "overall": overall.to_dicts(),
        "text_vector_dim": int(len(oof["text_vec"][0])),
        "params": {k: v for k, v in t.items() if k != "train_final"},
    }
    tables = {"overall": overall, "folds": details["folds"], "by_market": metrics_by(oof, "market")}
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
        write_report(reports / "text_model.md", summary, tables)
    (reports / "text_model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if t["train_final"]:
        log.info("Training the final model on all rows (%d epochs)", final_epochs)
        train_final(df, cfg, final_epochs, Path(paths["models"]) / "text_model")
    m = price_metrics(y, pred)
    print(
        f"Text model: MAE ${m['mae_usd']}, MdAPE {m['mdape_pct']}%, within 10% "
        f"{m['within_10_pct']}%. Report: {reports / 'text_model.md'}"
    )
    return summary


def log_results(cfg: dict[str, Any]) -> None:
    """Record a finished run (e.g. from Colab) in the local MLflow database."""
    reports = Path(cfg["paths"]["reports"])
    summary = json.loads((reports / "text_model_metrics.json").read_text(encoding="utf-8"))
    text_row = next(r for r in summary["overall"] if r["model"] == "Text (DistilBERT)")
    with mlflow_run(cfg, "text_distilbert") as mlflow:
        if mlflow is None:
            print("MLflow is disabled in config.yaml; nothing logged.")
            return
        mlflow.log_params({f"text_{k}": v for k, v in summary["params"].items()})
        mlflow.log_params({"final_epochs": summary["final_epochs"], "rows": summary["rows"]})
        mlflow.log_metrics({f"cv_{k}": v for k, v in text_row.items() if k != "model"})
        for p in (
            reports / "text_model.md",
            reports / "text_model_metrics.json",
            Path(cfg["paths"]["predictions"]) / "text_oof.parquet",
        ):
            if p.exists():
                mlflow.log_artifact(str(p))
    print("Logged the text model run to MLflow (experiment: airbnb-price, run: text_distilbert).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 text model")
    parser.add_argument("step", choices=["train", "log"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-mlflow", action="store_true", help="(train) skip MLflow")
    parser.add_argument("--epochs", type=int, help="override text_model.epochs")
    parser.add_argument("--no-final", action="store_true", help="skip the final all-rows model")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["text_model"]["epochs"] = args.epochs
    if args.no_final:
        cfg["text_model"]["train_final"] = False
    if args.step == "train":
        if args.no_mlflow:
            cfg["mlflow"]["enabled"] = False
        run(cfg)
        if cfg["mlflow"]["enabled"]:
            log_results(cfg)
    else:
        log_results(cfg)


if __name__ == "__main__":
    main()
