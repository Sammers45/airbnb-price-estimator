"""Phase 6: fusion network combining table, text and photo (runs on a normal CPU).

Usage: 
    python -m src.fusion train      # concat + attention, all inputs
    python -m src.fusion train --arch attention --modalities tab,text  # e.g. for ablations
    
Inputs (all out-of-fold, from earlier phases, joined on listing_id):
    data/processed/listings.parquet     table features, CLIP photot vectors   (Phase 1)
    data/predictions/baseline_oof.parquet       LightGBM prediction     (Phase 3)
    data/predictions/text_oof.parquet       768-d text vectors  (Phase 4)
    data/predictions/image_oof.parquet      photo prompt scores (Phase 5)
    
Tow architectures:
    concat      one 64-d block per input, joined, then an MLP.
    attention   each input is a token; one self-attention layer lets them interact, then
                attention pooling learns per listing how much to trust each input. A 
                missing photo is masked out, and the weights are saved for inspection.
                
Residual on the baseline: the network outputs a correction to the LightGBM prediction and 
starts at exactly zero, so the question it answers is "do text and photos improve on it?".

Stacking caveat: the text vectors and baseline predictions used as inputs are out-of-fold,
but each came from a model trained on the other four folds. This is standard stacking; it 
can make fusion scores very slightly optimistic.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

from src.baseline import FeatureBuilder, drop_outliers
from src.data import md
from src.image_model import image_matrix
from src.metrics import metrics_by, price_metrics
from src.utils import load_config, mlflow_run, set_seed, setup_logging

log = logging.getLogger(__name__)

MODALITIES = ("tab", "text", "image")

## ----------------------------------------------------------------------- INPUTS
def load_inputs(cfg: dict[str, Any]) -> pl.DataFrame:
    """Join the processed table with the out-of-fold outputs of Phases 3-5."""
    paths = cfg["paths"]
    preds = Path(paths["predictions"])
    needed = {
        "baseline_oof.parquet": "python -m src.baseline",
        "text_oof.parquet": "the Phase 4 Colab notebook",
        "image_oof.parquet": "python -m src.image_model train"
    }
    missing = [f"{f}  <- run {how}" for f, how in needed.items() if not (preds / f).exists()]
    if missing:
        raise FileNotFoundError("Missing inputs in data/predictions:\n  " + "\n".join(missing))
    df = pl.read_parquet(paths["processed"])
    base = pl.read_parquet(preds / "baseline_oof.parquet").select(
        "listing_id", pl.col("pred").log1p().alias("base_log")
    )
    text = pl.read_parquet(preds / "text_oof.parquet").select("listing_id", "text_vec")
    image = pl.read_parquet(preds / "image_oof.parquet")
    image = image.select("listing_id", *[c for c in image.columns if c.startswith("img_")])
    out = df.join(base, on="listing_id").join(text, on="listing_id").join(image, on="listing_id")
    if out.height != df.height:
        raise ValueError(
            f"Only {out.height} of {df.height} listings found in all prediction files;"
            "rerun the phase whose file is out of date."
        )
    return out.sort("listing_id")

@dataclass
class TabPreprocessor:
    """Table features for a neural network. Fit on training rows only.
    
    Numeric columns: missing values -> training median (plus a 0/1 "was missing" column),
    then standardised. Categories -> integer ids (0 = unseen or missing) for embeddings.
    """

    builder: dict[str, Any] = field(default_factory=dict)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    medians: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    flag_cols: list[int] = field(default_factory=list)
    cat_sizes: list[int] = field(default_factory=list)

    def fit(self, df: pl.DataFrame) -> TabPreprocessor:
        fb = FeatureBuilder().fit(df)
        self.builder = fb.to_dict()
        self.categorical = fb.categorical_names
        self.numeric = [c for c in fb.feature_names if c not in self.categorical]
        num = fb.transform(df).select(self.numeric).to_numpy().astype(np.float64)
        num = np.where(np.isfinite(num), num, np.nan)

        # Median of the observed values; a column with no values at all gets 0.
        self.medians = [
            float(np.median(c[~np.isnan(c)])) if (~np.isnan(c)).any() else 0.0 for c in num.T
        ]
        self.flag_cols = [i for i in range(num.shape[1]) if np.isnan(num[:, i]).any()]
        filled = self._fill(num)
        self.means = filled.mean(axis=0).tolist()
        self.stds = np.where(filled.std(axis=0) > 1e-9, filled.std(axis=0),
                             1.0).tolist()
        self.cat_sizes = [len(v) + 1 for v in fb.categories.values()]
        return self

    def _fill(self, num: np.ndarray) -> np.ndarray:
        num = np.where(np.isfinite(num), num, np.nan)
        return np.where(np.isnan(num), np.array(self.medians), num)

    def transform(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(numeric float32 matrix, categorical int64 matrix) for any rows."""
        fb = FeatureBuilder.from_dict(self.builder)
        x = fb.transform(df)
        num = x.select(self.numeric).to_numpy().astype(np.float64)
        flags = np.isnan(np.where(np.isfinite(num), num, np.nan))[:, self.flag_cols]
        num = (self._fill(num) - np.array(self.means)) / np.array(self.stds)
        cat = x.select(self.categorical).to_numpy()
        cat = np.where(np.isnan(cat), -1, cat).astype(np.int64) + 1  # unseen/missing -> 0
        return np.hstack([num, flags]).astype(np.float32), cat

    @property
    def n_numeric(self) -> int:
        return len(self.numeric) + len(self.flag_cols)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TabPreprocessor:
        return cls(**d)


@dataclass
class Arrays:
    """ALl model inputs for every listing, as tensors."""

    num: torch.Tensor
    cat: torch.Tensor
    text: torch.Tensor
    img: torch.Tensor
    has_img: torch.Tensor
    base: torch.Tensor
    y: torch.Tensor

    def take(self, idx: np.ndarray) -> Arrays:
        i = torch.as_tensor(idx, dtype=torch.long)
        return Arrays(*(getattr(self, f)[i] for f in self.__dataclass_fields__))


def build_arrays(
        df: pl.DataFrame,
        tab: TabPreprocessor,
        prompt_stats: tuple[np.ndarray, np.ndarray],
        mu: float,
        sd: float,
    ) -> Arrays:
    """Turn the joined frame into standardised tensors (statistics come from training rows)."""
    num, cat = tab.transform(df)
    text = np.array(df["text_vec"].to_list(), dtype=np.float32)
    img_vec, has = image_matrix(df)
    prompt_cols = [c for c in df.columns if c.startswith("img_")]
    if prompt_cols:
        p = df.select(prompt_cols).to_numpy().astype(np.float32)
        p = np.nan_to_num((p - prompt_stats[0]) / prompt_stats[1])
        img_vec = np.hstack([img_vec, p * has[:, None]])  
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)

    return Arrays(
        num=t(num),
        cat=t(cat, torch.long),
        text=t(text),
        img=t(img_vec),
        has_img=t(has.astype(np.float32)),
        base=t((df["base_log"].to_numpy() - mu) / sd),
        y=t((df["log_price"].to_numpy() - mu)/ sd),
    )

## -------------------------------------------------------------------------------- MODEL
def branch(d_in: int, hidden: int, dropout: float) -> nn.Sequential:
    """Project one input to ``hidden`` numbers."""
    return nn.Sequential(
        nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU()
    )


class FusionNet(nn.Module):
    """Concatenation or attention fusion over the chosen inputs, residual on the baseline."""

    def __init__(self, 
                 arch: str,
                 modalities: list[str],
                 n_numeric: int,
                 cat_sizes: list[int],
                 d_text: int,
                 d_img: int,
                 hidden: int = 64,
                 heads: int = 4,
                 dropout: float = 0.2,
                 residual: bool = True,
                ) -> None:
                    super().__init__()
                    if arch not in ("concat", "attention"):
                        raise ValueError("arch must be 'concat' or 'attention'")
                    if "tab" not in modalities:
                        raise ValueError("the table input ('tab') is always required")
                    self.arch, self.modalities, self.residual = arch, list(modalities), residual
                    self.embs = nn.ModuleList(nn.Embedding(n, min(16, (n + 1) // 2)) for n in cat_sizes)
                    d_tab = n_numeric + sum(e.embedding_dim for e in self.embs) + (1 if residual else 0)
                    self.tab = branch(d_tab, hidden, 0.0)
                    self.text = branch(d_text, hidden, dropout) if "text" in modalities else None
                    self.img = branch(d_img, hidden, dropout) if "image" in modalities else None
                    k = len(self.modalities)
                    if arch == "concat":
                         extra = 1 if "image" in modalities else 0
                         self.head = nn.Sequential(
                              nn.Linear(hidden * k + extra, hidden),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden, 1),
                         ) 
                    else:
                        self.mod_emb = nn.Parameter(torch.randn(k, hidden) * 0.02)
                        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
                        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
                        self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                        self.ff = nn.Sequential(
                             nn.Linear(hidden, 2 * hidden),
                             nn.GELU(),
                             nn.Dropout(dropout),
                             nn.Linear(2 * hidden, hidden),
                        )
                        self.pool = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                        self.head = nn.Sequential(
                             nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
                        )
                    if residual:    # start exactly at the baseline prediction
                        nn.init.zeros_(self.head[-1].weight)
                        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, a: Arrays, drop_text: torch.Tensor | None = None, drop_img: torch.Tensor | None = None 
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
         """Returns (standardized log-price prediction, per-input attention weights or None)."""
         n = a.num.shape[0]
         parts = [a.num] + [e(a.cat[:, i]) for i, e in enumerate(self.embs)]
         if self.residual:
            parts.append(a.base[:, None])
         tokens = [self.tab(torch.cat(parts, dim=1))]
         missing = [torch.zeros(n, dtype=torch.bool)]
         if self.text is not None:
              tokens.append(self.text(a.text))
              missing.append(drop_text if drop_text is not None else torch.zeros(n, dtype=torch.bool))
         if self.img is not None:
              tokens.append(self.img(a.img))
              gone = a.has_img < 0.5
              missing.append(gone | drop_img if drop_img is not None else gone)
         mask = torch.stack(missing, dim=1)

         if self.arch == "concat":
              blocks = [t * (~mask[:, i : i + 1]).float() for i, t in enumerate(tokens)] 
              if self.img is not None:
                   blocks.append((~mask[:, -1:]).float())
              out, weights = self.head(torch.cat(blocks, dim=1)).squeeze(-1), None
         else:
              x = torch.stack(tokens, dim=1) + self.mod_emb[None]
              h = self.ln1(x)
              x = x + self.self_attn(h, h, h, key_padding_mask=mask, need_weights=False)[0]
              x = x + self.ff(self.ln2(x))
              pooled, w = self.pool(self.cls.expand(n, -1, -1), x, x, key_padding_mask=mask)
              out, weights = self.head(pooled.squeeze(1)).squeeze(-1), w.squeeze(1)
         return (a.base + out if self.residual else out), weights


## ----------------------------------------------------------------------- Training
def new_net(
          cfg: dict, arch: str, modalities: list[str], tab: TabPreprocessor, a: Arrays
) -> FusionNet:
    f = cfg["fusion"]
    return FusionNet(
         arch, 
         modalities,
         tab.n_numeric,
         tab.cat_sizes,
         a.text.shape[1],
         a.img.shape[1],
         f["hidden"],
         f["heads"],
         f["dropout"],
         f["residual_on_baseline"],
    )


def predict(net: FusionNet, a: Arrays) -> tuple[np.ndarray, np.ndarray | None]:
    net.eval()
    with torch.no_grad():
        out, w = net(a)
    return out.numpy(), (w.numpy() if w is not None else None)

def fit_net(
        net: FusionNet,
        train: Arrays,
        check: Arrays | None,
        cfg: dict,
        seed: int,
        epochs: int | None = None,
    ) -> tuple[FusionNet, int]:
    """Mini-batch training; with ``check`` rows, stop early and keep the best epoch."""
    f = cfg["fusion"]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(
         net.parameters(), lr=float(f["learning_rate"]), weight_decay=float(f["weight_decay"])
    )
    max_epochs = epochs or f["max_epochs"]
    best, best_epoch, best_state, waited = math.inf, max_epochs, None, 0
    n = train.y.shape[0]
    for epoch in range(1, max_epochs + 1):
        net.train()
        for idx in torch.randperm(n, generator=gen).split(f["batch_size"]):
             b = train.take(idx.numpy())
             p = float(f["modality_dropout"])
             drop_t = torch.rand(len(idx), generator=gen) < p
             drop_i = torch.rand(len(idx), generator=gen) < p
             pred, _ = net(b, drop_t, drop_i)
             loss = nn.functional.mse_loss(pred, b.y)
             opt.zero_grad()
             loss.backward()
             nn.utils.clip_grad_norm_(net.parameters(), 1.0)
             opt.step()
        if check is not None:
             mse = float(np.mean((predict(net, check)[0] - check.y.numpy()) ** 2))
             if mse < best - 1e-5:
                  best, best_epoch, waited = mse, epoch, 0
                  best_state = copy.deepcopy(net.state_dict())
             else:
                  waited += 1
                  if waited >= f["patience"]:
                       break
    if best_state is not None:
         net.load_state_dict(best_state)
    return net, best_epoch


def prompt_stats(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the photo prompt scores on the given (training) rows."""
    cols = [c for c in df.columns if c.startswith("img_")]
    if not cols:
         return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    p = df.select(cols).to_numpy().astype(np.float64)
    mean, std = np.nanmean(p, axis=0), np.nanstd(p, axis=0)
    return mean.astype(np.float32), np.where(std > 1e-9, std, 1.0).astype(np.float32)

def cross_validate(
          df: pl.DataFrame, cfg: dict, arch: str, modalities: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
     """Out-of-fold predictions (mean of seeds) and attention weights for every
     listing."""
     f = cfg["fusion"]
     folds = df["fold"].to_numpy()
     pred_log = np.zeros(df.height)
     weights = np.full((df.height, len(modalities)), np.nan)
     rows = []
     for k in sorted(np.unique(folds)):
          train_df = drop_outliers(df.filter(pl.col("fold") != k), cfg)
          valid_idx = np.where(folds == k)[0]
          mu, sd = float(train_df["log_price"].mean()), float(train_df["log_price"].std())
          tab = TabPreprocessor().fit(train_df)
          stats = prompt_stats(train_df)
          train_a = build_arrays(train_df, tab, stats, mu, sd)
          valid_a = build_arrays(df, tab, stats, mu, sd).take(valid_idx)
          inner = (
               np.random.default_rng(cfg["seed"] + int(k)).random(train_df.height)
               < f["inner_valid_frac"]
          )
          preds, ws, epochs = [], [], []
          for s in range(f["seeds"]):
            seed = cfg["seed"] + 100 * int(k) + s
            set_seed(seed)
            net = new_net(cfg, arch, modalities, tab, train_a)
            net, best_epoch = fit_net(
                 net, train_a.take(np.where(~inner)[0]), train_a.take(np.where(inner)[0]), cfg, seed
            )
            p, w = predict(net, valid_a)
            preds.append(p * sd + mu)
            epochs.append(best_epoch)
            if w is not None:
                 ws.append(w)
          pred_log[valid_idx] = np.mean(preds, axis=0)
          if ws:
            weights[valid_idx] = np.mean(ws, axis=0)
          m = price_metrics(df["price"].to_numpy()[valid_idx],
                       np.expm1(pred_log[valid_idx]))
          rows.append({"fold": int(k), "train_rows": int((~inner).sum()), "best_epochs": str(epochs), **m})
          log.info("%s fold %d: MdAPE %.2f%%, within 10%% %.1f%%", arch, k, m["mdape_pct"], m["within_10_pct"])
     oof = df.select("listing_id", "fold", "market", "room_type", "price").with_columns(
        pl.Series("pred_log", pred_log),
        pl.Series("pred", np.expm1(pred_log)),
        pl.Series("has_image", df["has_image_emb"].to_numpy()),
    )
     if arch == "attention":
         oof = oof.with_columns(pl.Series(f"w_{m}", weights[:, i]) for i, m in enumerate(modalities))
     return oof, pl.DataFrame(rows)


def train_final(
          df: pl.DataFrame, cfg: dict, arch: str, modalities: list[str], epochs: int, out: Path 
) -> None:
     """One model on all rows (fixed number of epochs from CV) for the web app."""
     train_df = drop_outliers(df, cfg)
     mu, sd = float(train_df["log_price"].mean()), float(train_df["log_price"].std())
     tab = TabPreprocessor().fit(train_df)
     stats = prompt_stats(train_df)
     a = build_arrays(train_df, tab, stats, mu, sd)
     set_seed(cfg["seed"])
     net, _ = fit_net(
          new_net(cfg, arch, modalities, tab, a), a, None, cfg, cfg["seed"], epochs=epochs
     )
     out.mkdir(parents=True, exist_ok=True)
     torch.save(net.state_dict(), out / "model.pt")
     meta = {
          "arch": arch,
          "modalities": modalities,
          "target_mean": mu,
          "target_std": sd,
          "epochs": epochs,
          "prompt_mean": stats[0].tolist(),
          "prompt_std": stats[1].tolist(),
          "d_text": int(a.text.shape[1]),
          "d_img": int(a.img.shape[1]),
          "tab": tab.to_dict(),
          "fusion": cfg["fusion"],
     }
     (out / "fusion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


## ----------------------------------------------------------------------------------- REPORT
LABELS = {"concat": "Fusion: concatenation", "attention": "Fusion: attention"}

def model_name(arch: str, modalities: list[str]) -> str:
     extra = "" if modalities == list(MODALITIES) else f" ({'+'.join(modalities)})"
     return LABELS[arch] + extra

def comparison(df: pl.DataFrame, results: dict[str, pl.DataFrame], cfg: dict) -> pl.DataFrame:
     """Every model on the same listings, with the change in MdAPE vs the baseline."""
     y = df["price"].to_numpy()
     rows = [{"model": "LightGBM baseline", **price_metrics(y,
    np.expm1(df["base_log"].to_numpy()))}]
     text = pl.read_parquet(Path(cfg["paths"]["predictions"]) / "text_oof.parquet").sort(
        "listing_id"
     ) 
     rows.append(
        {
            "model": "Text only (DistilBERT)",
            **price_metrics(text["price"].to_numpy(), text["pred"].to_numpy()),
        }
     )
     for name, oof in results.items():
        rows.append(
            {"model": name, **price_metrics(oof["price"].to_numpy(),
            oof["pred"].to_numpy())}
        )
     out = pl.DataFrame(rows)
     return out.with_columns(
        (pl.col("mdape_pct") - out["mdape_pct"][0]).round(2).alias("mdape_vs_baseline")
     )


def attention_tables(oof: pl.DataFrame, modalities: list[str]) -> dict[str, pl.DataFrame]:
    w = [pl.col(f"w_{m}").mean().round(3).alias(m) for m in modalities]
    overall = oof.select(pl.lit("all listings").alias("group"), *w)
    by_photo = (
        oof.group_by("has_image")
        .agg(*w)
        .sort("has_image")
        .with_columns(
            pl.when(pl.col("has_image") == 1)
            .then(pl.lit("with_photo"))
            .otherwise(pl.lit("no photo"))
            .alias("group")
        )
        .select("group", *modalities)
    )
    by_market = oof.group_by("market").agg(pl.len().alias("rows"), *w).sort("rows",
    descending=True)
    return {"overall": pl.concat([overall, by_photo]), "by_market": by_market} 


def write_report(path: Path, s: dict, t: dict[str, Any]) -> None:
    lines = [
        "# Fusion: table + text + photo",
        "_Generated by `python -m src.fusion`. Do not wdit by hand._",
        f"\n{s['rows']} listings, phase 1 folds, {s['seeds']} seeds averaged per fold. Inputs: "
        f"{', '.join(s['modalities'])}. Residual on the LightGBMM baseline: {s['residual']}."
        "Metrics are out-of-fold, in USD, on the same listings for every model.",
        "\n## 1. Comparison",
        md(t["comparison"]),
        "\n`mdape_vs_baseline` < 0 means the model is better than the baseline. Phase 7 tests "
        "whether each gain is statistically real (paired bootstrap on per-listing errors)."
    ]
    for i, (name, folds) in enumerate(t["folds"].items(), start=2):
        lines += [f"\n## {i}. Per fold: {name}", md(folds)]
    n = len(t["folds"]) + 2
    lines += [f"\n## {n}. Per market: baseline vs best fusion ({s['best']})",
    md(t["by_market"])]
    if "attention" in t:
        lines += [
            f"\n## {n + 1}. What the attention model relied on",
            "Average attention weight per input (sums to 1 per listing). With no photo, the "
            "photo weight is 0 by design (masked).",
            md(t["attention"]["overall"]),
            "\nBy Market:",
            md(t["attention"]["by_market"]),
        ]
    lines += [
        f"\n## {n + 2}. Notes",
        "- Stacking: text vectors and the baseline prediction are out-of-fold inputs, but each "
        "came from models trained on the other folds. This is standard practice and can make "
        "fusion scores very slightly optimistic.",
        f"- Final models (all rows, {s['final_epochs']} epochs) are saved in `models/fusion_*` "
        "for the web app.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any], archs: list[str], modalities: list[str]) -> dict[str, Any]:
     set_seed(cfg["seed"])
     torch.set_num_threads(max(1, torch.get_num_threads()))
     paths = cfg["paths"]
     reports, pred_dir = Path(paths["reports"]), Path(paths["predictions"])
     reports.mkdir(parents=True, exist_ok=True)
     df = load_inputs(cfg)
     log.info("Fusion on %d listings; arch=%s; inputs=%s", df.height, archs, modalities)
     tag = "" if modalities == list(MODALITIES) else "_" + "_".join(modalities)
     results, folds, final_epochs = {}, {}, {}
     for arch in archs:
          oof, fold_table = cross_validate(df, cfg, arch, modalities)
          oof.write_parquet(pred_dir / f"fusion_{arch}{tag}_oof.parquet")
          name = model_name(arch, modalities)
          results[name], folds[name] = oof, fold_table
          epochs = [e for row in fold_table["best_epochs"] for e in json.loads(row)]
          final_epochs[arch] = max(1, int(np.median(epochs)))
          if cfg["fusion"]["train_final"] and not tag:
               train_final(
                    df, 
                    cfg,
                    arch,
                    modalities,
                    final_epochs[arch],
                    Path(paths["models"]) / f"fusion_{arch}",
               )

     comp = comparison(df, results, cfg)
     fused = comp.filter(pl.col("model").str.starts_with("Fusion"))
     best = fused.sort("mdape_pct").row(0, named=True)["model"]
     best_oof = results[best]
     base_m = metrics_by(
          df.select("market", "price", pl.col("base_log").exp().sub(1).alias("pred")),
          "market"
     )
     fus_m = metrics_by(best_oof, "market")
     by_market = (
          base_m.select("market", "rows", pl.col("mdape_pct").alias("baseline_mdape"))
          .join(fus_m.select("market", pl.col("mdape_pct").alias("fusion_mdape")),
        on = "market")
          .with_columns((pl.col("fusion_mdape") - pl.col("baseline_mdape")).round(2).alias("change"))
          .sort("rows", descending=True)
     )

     tables: dict[str, Any] = {"comparison": comp, "folds": folds, "by_market": by_market}
     att = next((n for n in results if n.startswith(LABELS["attention"])), None)
     if att:
          tables["attention"] = attention_tables(results[att], modalities)
     summary = {
          "rows": df.height,
          "seeds": cfg['fusion']['seeds'],
          "modalities": modalities,
          "residual": cfg['fusion']['residual_on_baseline'],
          "best": best,
          "final_epochs": final_epochs,
          "comparison": comp.to_dicts(),
     }
     name = f"fusion{tag}"
     with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
          write_report(reports / f"{name}.md", summary, tables)
     (reports / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")

     for arch_name, oof in results.items():
          arch = "attention" if arch_name.startswith(LABELS["attention"]) else "concat"
          with mlflow_run(cfg, f"fusion_{arch}{tag}") as mlflow:
               if mlflow is not None:
                    mlflow.log_params({f"fusion_{k}": v for k, v in cfg["fusion"].items()})
                    mlflow.log_params({"arch": arch, "modalities": ",".join(modalities)})
                    m = price_metrics(oof["price"].to_numpy(), oof["pred"].to_numpy())
                    mlflow.log_metrics({f"cv_{k}": v for k, v in m.items()})
                    mlflow.log_artifact(str(reports / f"{name}.md"))

     for row in comp.iter_rows(named=True):
          print(
               f"{row['model']:<40} MdAPE {row['mdape_pct']:>6}% "
               f"within 10% {row['within_10_pct']:>6}%  MAE ${row['mae_usd']}"
          )
     print(f"Report: {reports / f'{name}.md'}")
     return summary


def main() -> None:
     parser = argparse.ArgumentParser(description="Phase 6 fusion network")
     parser.add_argument("step", choices=["train"])
     parser.add_argument("--config", default="config.yaml")
     parser.add_argument("--arch", choices=["concat", "attention", "both"], default="both")
     parser.add_argument("--modalities", default="tab,text,image", help="e.g. tab,text")
     args = parser.parse_args()
     setup_logging()
     cfg = load_config(args.config)
     archs = ["concat", "attention"] if args.arch == "both" else [args.arch]
     mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
     unknown = set(mods) - set(MODALITIES)
     if unknown:
          parser.error(f"unknown modalities {sorted(unknown)}; choose from {MODALITIES}")
     run(cfg, archs, [m for m in MODALITIES if m in mods])


if __name__ == "__main__":
     main()
