"""Phase 8b: why does the model predict this price? Runs on a CPU

Usage:
    python -m src.explain

1. SHAP for the Phase 3 LightGBM baseline. LigthGBM computes exact TreeSHAP values itself
   (``pred_contrib=True``), the same algorithm as the ``shap`` package. On the log scale, the values
   add up exactly to the prediction, so each one is a price multipler:
   effect % = exp(value) - 1.
2. Integrated Gradients (Captum) for the Phase 6 fusion network (concatenation). The 
network predicts a CORRECTION to the LightGBM price; IG splits that correction between the table, the description 
and the photo, relative to an "average listing" baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.baseline import FeatureBuilder
from src.data import md
from src.utils import load_config, mlflow_run, setup_logging, set_seed

log = logging.getLogger(__name__)

## ------------------------------------------------------------------------------ SHAP
def load_baseline(models: Path) -> tuple[lgb.Booster, FeatureBuilder]:
    path = models / "baseline_lightgbm.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run Phase 3 first (python -m src.baseline)")
    fb = json.loads((models / "baseline_features.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(path)), FeatureBuilder.from_dict(fb)

def shap_values(booster: lgb.Booster, x: np.ndarray) -> tuple[np.ndarray, float]:
    """(per-feature SHAP values on the log scale, expected value.)"""
    contrib = booster.predict(x, pred_contrib=True)
    return contrib[:, :-1], float(contrib[0, -1])

def effect_pct(v: np.ndarray | float) -> np.ndarray | float:
    """Log-Scale SHAP value -> % change in (price + 1)."""
    return 100 * (np.exp(v) - 1)

def show_value(fb: FeatureBuilder, name: str, v: float) -> str:
    if np.isnan(v):
        return "missing"
    if name in fb.categories:
        return fb.categories[name][int(v)]
    return f"{v:g}" if abs(v) >= 0.01 else f"{v:.3g}"

def explain_listing(
        booster: lgb.Booster, fb: FeatureBuilder, row: pl.DataFrame, top: int = 6
) -> dict[str, Any]:
    """Explanation of one listing."""
    x = fb.matrix(row)
    phi, base = shap_values(booster, x)
    order = np.argsort(-np.abs(phi[0]))[:top]
    return {
        "typical_price": round(float(np.expm1(base)), 2),
        "predicted_price": round(float(np.expm1(base + phi[0].sum())), 2),
        "reasons": [
            {
                "feature": fb.feature_names[j],
                "value": show_value(fb, fb.feature_names[j], x[0, j]),
                "effect_pct": round(float(effect_pct(phi[0, j])), 1),
            }
            for j in order
        ],
    }

def shap_table(phi: np.ndarray, x: np.ndarray, fb: FeatureBuilder) -> pl.DataFrame:
    """Global importance: average size of each feature's effect, and its direction."""
    rows = []
    for j, name in enumerate(fb.feature_names):
        mean_abs = float(np.abs(phi[:, j]).mean())
        if name in fb.categories:
            direction = "depends on the category"
        else:
            ok = ~np.isnan(x[:, j])            
            rho = (
                np.corrcoef(np.argsort(np.argsort(x[ok, j])), phi[ok, j])[0, 1]
                if ok.sum() > 2 and np.std(x[ok, j]) > 0 and np.std(phi[ok, j]) > 0
                else 0.0
            )
            direction = (
                "higher = pricier" if rho > 0.3 else "higher = cheaper" if rho < -0.3 else "mixed"
            )
        rows.append(
            {
                "feature": name,
                "mean_abs_shap": round(mean_abs, 4),
                "typical_effect_pct": round(float(effect_pct(mean_abs)), 1),
                "direction": direction,
            }
        )
    return pl.DataFrame(rows).sort("mean_abs_shap", descending=True)

def plot_shap(phi: np.ndarray, x: np.ndarray, names: list[str], top: int, path: Path) -> str:
    """Beeswarm-style chart: one dot per listing, colour = feature value (low -> high)."""
    order = np.argsort(-np.abs(phi).mean(axis=0))[:top][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 0.4 * top + 1.5))
    for y, j in enumerate(order):
        v = x[:, j]
        ok = ~np.isnan(v)
        colour = np.full(len(v), 0.5)
        colour[ok] = np.argsort(np.argsort(v[ok])) / max(ok.sum() - 1, 1)
        ax.scatter(
            effect_pct(phi[:, j]),
            y + rng.uniform(-0.3, 0.3, len(v)),
            c = colour,
            cmap = "coolwarm",
            s = 4,
            alpha = 0.6,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)), [names[j] for j in order])
    ax.set(
        xlabel = "Effect on price, % (red = high feature value, blue = low)",
        title = "What drives the LightGBM price (SHAP)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"figures/{path.name}"

## --------------------------------------------------------------------------------------- Integrated Gradients
def integrated_gradients(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Split the fusion network's correction between table, text and photo. None if not trained."""
    model_dir = Path(cfg["paths"]["models"]) / "fusion_concat"
    if not (model_dir / "model.pt").exists():
        log.warning("No fusion model in %s: skipping Integrated Gradients (run Phase 6)", model_dir)
        return None
    import torch
    from captum.attr import IntegratedGradients

    from src import fusion as fu

    meta = json.loads((model_dir / "fusion.json").read_text(encoding="utf-8"))
    tab = fu.TabPreprocessor.from_dict(meta["tab"])
    df = fu.load_inputs(cfg)
    stats = (np.array(meta["prompt_mean"], np.float32), np.array(meta["prompt_std"], np.float32))
    a = fu.build_arrays(df, tab, stats, meta["target_mean"], meta["target_std"])
    net = fu.new_net({**cfg, "fusion": meta["fusion"]}, meta["arch"], meta["modalities"], tab, a)
    net.load_state_dict(torch.load(model_dir / "model.pt"))
    net.eval()

    e = cfg["explain"]
    rows = np.random.default_rng(cfg["seed"]).permutation(df.height)[: e["ig_rows"]]
    s = a.take(rows)

    def forward(num, text, img, cat, has_img, base): 
        return net(fu.Arrays(num, cat, text, img, has_img, base, base))[0]

    inputs = (s.num, s.text, s.img)
    baselines = tuple(
        t.mean(dim=0, keepdim=True) for t in (a.num, a.text, a.img)
    ) 
    attr, delta = IntegratedGradients(forward).attribute(
        inputs,
        baselines=baselines,
        additional_forward_args=(s.cat, s.has_img, s.base),
        n_steps=e["ig_steps"],
        return_convergence_delta=True
    )
    sd = meta["target_std"] # standardized -> log-price units
    # Each input's net push on the price 
    groups = ("table", "text", "photo")
    size = {
        m: (t.sum(dim=1).abs() * sd).detach().numpy() for m, t in zip(groups, attr, strict=True)
    }
    total = sum(size.values())
    has = s.has_img.numpy() > 0.5

    def shares(mask: np.ndarray) -> dict[str, float]:
        t = total[mask].sum()
        return {m: round(100 * float(v[mask].sum() / t), 1) for m, v in size.items()}

    names = tab.numeric + [f"{tab.numeric[i]} missing" for i in tab.flag_cols]
    tab_attr = (attr[0].abs().mean(dim=0) * sd).detach().numpy()
    with torch.no_grad():
        start = [b.expand_as(i) for b, i in zip(baselines, inputs, strict=True)]
        rest = (s.cat, s.has_img, s.base)
        gap = forward(*inputs, *rest) - forward(*start, *rest)
    return {
        "rows": int(len(rows)),
        "steps": e["ig_steps"],
        "shares": pl.DataFrame(
            [
                {"listings": "all", "rows": int(len(rows)), **shares(np.ones(len(rows), bool))},
                {"listings": "with photo", "rows": int(has.sum()), **shares(has)},
                {"listings": "no photo", "rows": int((~has).sum()), **shares(~has)},
            ]
        ).rename({"table": "table_pct", "text": "text_pct", "photo": "photo_pct"}),
        "top_table": pl.DataFrame(
            {"feature": names, "mean_abs_effect_pct": np.round(effect_pct(tab_attr), 2)}
        )
        .sort("mean_abs_effect_pct", descending=True)
        .head(10),
        "correction_pct": round(float(effect_pct(np.abs(gap.numpy() * sd).mean())), 2),
        "delta_ratio": round(float(delta.abs().mean() / max(gap.abs().mean(), 1e-12)), 4),
    }


## --------------------------------------------------------------------------------- REPORT
def write_report(path: Path, s: dict[str, Any], t: dict[str, Any], fig: str) -> None:
    lines = [
        "# Explanations: why the model predicts this price1",
        "_Generated by `python -m src.explain`. Do not edit by hand_.",
        "",
        "## 1. What drives the price (SHAP, Phase 3 LightGBM)",
        f"{s['rows']} listings. A typical listing is priced at ${s['typical_price']}; each "
        "feature multiplies that up or down. `typical_effect_pct` is the average size of that "
        "multiplier. Check: SHAP values add up to the model's prediction "
        f"(max gap {s['shap_gap']:.1e}).",
        "",
        md(t["shap"].head(s["shap_top"])),
        "",
        f"![shap]({fig})",
        "",
        "## 2. Example listings, explained one by one",       
    ]
    for ex in t["examples"]:
        lines += [
            "",
            f"**{ex['title']}**: actual ${ex['actual']}, predicted ${ex['predicted_price']} "
            f"(typical ${ex['typical_price']})",
            "",
            md(pl.DataFrame(ex['reasons'])),
        ]
    lines += ["", "## 3. What the fusion network adds (Integrated Gradients, Captum)"]
    ig = t["ig"]
    if ig is None:
        lines.append("Skipped: no fusion model found (rum `python -m src.fusion train`).")
    else:
        lines += [
            f"{ig['rows']} random listings, {ig['steps']}, baseline = the average listing. "
            "The network starts from the LightGBM price and adds a correction; on average the "
            f"correction is {ig['correction_pct']}% of the price. SHare of the correction from "
            "each input:",
            "",
            md(ig["shares"]),
            "",
            "Table features with the largest effect on the correction:",
            "",
            md(ig["top_table"]),
            "",
            f"Completeness check: attributions add up to the change in output to within "
            f"{100 * ig['delta_ratio']:.2f}%. Categories (market, room type, ...) and the LigthGBM "
            "price are held fixed, so they get no share here.",
        ]
    lines += [
        "",
        "## 4. Notes",
        "- SHAP explains the baseline model on the listings it was trained on (global view); "
        "the web app shows the same explanation for each new listing.",
        "- Effects are multiplicative: +20% and -10% together give 1.2 x 0.9 = 1.08 (+8%).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg['seed'])
    paths, e = cfg["paths"], cfg["explain"]
    reports = Path(paths["reports"])
    booster, fb = load_baseline(Path(paths["models"]))
    df = pl.read_parquet(paths["processed"])
    x = fb.matrix(df)
    phi, base = shap_values(booster, x)
    gap = float(np.abs(base + phi.sum(axis=1) - booster.predict(x)).max())
    table = shap_table(phi, x, fb)
    fig = plot_shap(
        phi, x, fb.feature_names, e["shap_top"], reports / "figures" / "shap_summary.png"
    )

    pred = np.expm1(base + phi.sum(axis=1))
    order = np.argsort(pred)
    examples = []
    for title, q in (("Budget listing", 0.1), ("Typical listing", 0.5), ("Premium listing", 0.9))[
        : e["examples"]
    ]:
        i = int(order[int(q * (len(order) - 1))])
        ex = explain_listing(booster, fb, df[i : i + 1])
        examples.append({"title": title, "actual": round(float(df["price"][i]), 2), **ex})

    ig = integrated_gradients(cfg)
    summary = {
        "rows": df.height,
        "typical_price": round(float(np.expm1(base)), 2),
        "shap_gap": gap,
        "shap_top": e["shap_top"],
        "top_features": table.head(10).to_dicts(),
        "examples": examples,
        "ig": None
        if ig is None
        else {k: (v.to_dicts() if isinstance(v, pl.DataFrame) else v) for k, v in ig.items()},
    }
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):
        write_report(
            reports / "explain.md", summary, {"shap": table, "examples": examples, "ig": ig}, fig
        )
    (reports / "explain_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow_run(cfg, "explain") as mlflow:
        if mlflow is not None:
            mlflow.log_metric("shap_gap", gap)
            if ig is not None:
                for r in ig["shares"].filter(pl.col("listings") == "all").iter_rows(named=True):
                    for m in ("table", "text", "photo"):
                        mlflow.log_metric(f"ig_share{m}", r[f"{m}_pct"])
            mlflow.log_artifact(str(reports / "explain.md"))
            mlflow.log_artifact(str(reports / fig))

    print("Top features (SHAP):")
    for r in table.head(8).iter_rows(named=True):
        print(
            f"  {r['feature']:<26} typical effect {r['typical_effect_pct']:>5}% {r['direction']}"
        )
    if ig is not None:
        r = ig["shares"].row(0, named=True)
        print(
            f"Fusion correction ({ig['correction_pct']}% of price on average) comes from: "
            f"table {r['table_pct']}%, text {r['text_pct']}%, photo {r['photo_pct']}%"
        )
    print(f"Report: {reports / 'explain.md'}")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8b: SHAP and Integrated Gradients")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    run(load_config(args.config))


if __name__ == "__main__":
    main()