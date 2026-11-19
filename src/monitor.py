"""Phase 9: monitoring the live price service. Runs on a CPU.

Usage:
    python -m src.monitor                                   # report on logged requests
    python -m src.monitor simulate --n 300 --fresh          # replay real listings
    python -m src.monitor simulate --n 200 --market "Rio De Janeiro"   # shifted week: drift

Three kinds of checks, compared with the training data (models/serving/reference.json):
1. Pipeline health: errors, latency, how often the photo / description were used.
2. Drift: do incoming listings and predicted prices look like the training data?
   PSI (population stability index): < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 major shift.
3. Outcomes: once real prices come back (POST /feedback), the actual error and how often the
   80% range contained the real price.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.data import md
from src.utils import load_config, setup_logging

log = logging.getLogger(__name__)
EPS = 1e-4


# --------------------------------------------------------------------------- reference
def edges_of(values: np.ndarray, bins: int = 10) -> list[float]:
    """Inner bin edges at the deciles of the reference values (duplicates removed)."""
    v = values[~np.isnan(values)]
    return sorted(set(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]).round(6).tolist()))


def shares(values: np.ndarray, edges: list[float]) -> list[float]:
    """Share of values in each bin; missing values get their own last bin."""
    v = np.asarray(values, dtype=float)
    counts = np.bincount(np.digitize(v[~np.isnan(v)], edges), minlength=len(edges) + 1)
    counts = np.append(counts, np.isnan(v).sum())
    return (counts / max(len(v), 1)).tolist()


def cat_shares(values: list[Any], cats: list[str]) -> list[float]:
    """Share of each known category, plus one bin for anything unseen."""
    n = max(len(values), 1)
    known = [sum(v == c for v in values) / n for c in cats]
    return [*known, 1 - sum(known)]


def psi(ref: list[float], live: list[float]) -> float:
    r, x = np.clip(np.array(ref), EPS, None), np.clip(np.array(live), EPS, None)
    return float(np.sum((x - r) * np.log(x / r)))


def build_reference(cfg: dict[str, Any]) -> dict[str, Any]:
    """What 'normal' looks like: training feature mix, predicted prices, review rate, widths."""
    m = cfg["monitor"]
    df = pl.read_parquet(cfg["paths"]["processed"]).with_columns(pl.col("market").fill_null(""))
    ref: dict[str, Any] = {"numeric": {}, "categories": {}}
    for c in m["features"]:
        v = df[c].cast(pl.Float64).to_numpy()
        e = edges_of(v)
        ref["numeric"][c] = {"edges": e, "shares": shares(v, e)}
    for c in m["categories"]:
        cats = df[c].drop_nulls().unique().sort().to_list()
        ref["categories"][c] = {"cats": cats, "shares": cat_shares(df[c].to_list(), cats)}
    preds = Path(cfg["paths"]["predictions"])
    iv = preds / "intervals_oof.parquet"
    if iv.exists():
        o = pl.read_parquet(iv)
        p = o["pred"].to_numpy()
        e = edges_of(p)
        k = f"{round(cfg['intervals']['coverage_levels'][0] * 100)}"
        ref["price"] = {"edges": e, "shares": shares(p, e)}
        ref["review_pct"] = round(100 * float(o["review"].mean()), 2)
        ref["median_width_pct"] = round(
            100 * float(((o[f"hi_{k}"] - o[f"lo_{k}"]) / o["pred"]).median()), 1
        )
        ref["coverage_pct"] = round(
            100 * float(((o["price"] >= o[f"lo_{k}"]) & (o["price"] <= o[f"hi_{k}"])).mean()), 2
        )
    else:
        log.warning(
            "%s not found: price / review-rate drift checks will be skipped (run Phase 8)", iv
        )
    out = Path(cfg["paths"]["models"]) / "serving" / "reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    return ref


# --------------------------------------------------------------------------- summary
def read_log(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    return pl.read_ndjson(path, infer_schema_length=None)


def status(value: float, m: dict[str, Any]) -> str:
    return (
        "major shift"
        if value > m["psi_major"]
        else "moderate shift"
        if value > m["psi_moderate"]
        else "stable"
    )


def summarise(cfg: dict[str, Any]) -> dict[str, Any]:
    """Everything the monitoring report and the /monitoring endpoint show."""
    m, logs = cfg["monitor"], Path(cfg["serve"]["log_dir"])
    ref_path = Path(cfg["paths"]["models"]) / "serving" / "reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    req, fb = read_log(logs / "requests.jsonl"), read_log(logs / "feedback.jsonl")
    n_err = len(read_log(logs / "errors.jsonl"))
    out: dict[str, Any] = {"requests": req.height, "alerts": []}
    if req.height == 0:
        return out

    # 1. pipeline health
    lat = req["latency_ms"].to_numpy()
    out["pipeline"] = {
        "requests": req.height,
        "errors": n_err,
        "error_pct": round(100 * n_err / (req.height + n_err), 2),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "with_photo_pct": round(100 * float(req["used_photo"].mean()), 1),
        "with_text_pct": round(100 * float(req["used_text"].mean()), 1),
        "review_pct": round(100 * float(req["review"].mean()), 1),
        "median_width_pct": round(
            100 * float(((req["high"] - req["low"]) / req["price"]).median()), 1
        ),
    }
    if out["pipeline"]["error_pct"] > m["max_error_pct"]:
        out["alerts"].append(
            f"Error rate {out['pipeline']['error_pct']}% (limit {m['max_error_pct']}%)"
        )
    if out["pipeline"]["latency_p95_ms"] > m["latency_p95_ms"]:
        out["alerts"].append(
            f"Slow: p95 latency {out['pipeline']['latency_p95_ms']} ms "
            f"(limit {m['latency_p95_ms']} ms)"
        )

    # 2. drift
    drift = []
    if ref is not None and req.height >= m["min_requests"]:  # too few requests = noisy PSI
        for c, r in ref["numeric"].items():
            if c in req.columns:
                drift.append(
                    (
                        "input",
                        c,
                        psi(r["shares"], shares(req[c].cast(pl.Float64).to_numpy(), r["edges"])),
                    )
                )
        for c, r in ref["categories"].items():
            if c in req.columns:
                drift.append(
                    ("input", c, psi(r["shares"], cat_shares(req[c].to_list(), r["cats"])))
                )
        if "price" in ref:
            drift.append(
                (
                    "prediction",
                    "predicted price",
                    psi(
                        ref["price"]["shares"],
                        shares(req["price"].to_numpy(), ref["price"]["edges"]),
                    ),
                )
            )
            gap = out["pipeline"]["review_pct"] - ref["review_pct"]
            if abs(gap) > m["review_rate_tolerance"]:
                out["alerts"].append(
                    f"Review rate {out['pipeline']['review_pct']}% "
                    f"vs {ref['review_pct']}% in training"
                )
    out["drift"] = [
        {"kind": k, "feature": f, "psi": round(v, 3), "status": status(v, m)} for k, f, v in drift
    ]
    out["alerts"] += [
        f"{d['status'].capitalize()} in {d['feature']} (PSI {d['psi']})"
        for d in out["drift"]
        if d["status"] != "stable"
    ]

    # 3. outcomes
    if fb.height:
        j = req.join(fb.select("id", "actual_price").unique("id", keep="last"), on="id")
        if j.height:
            y, p = j["actual_price"].to_numpy(), j["price"].to_numpy()
            ape = np.abs(p - y) / y
            cov = 100 * float(((y >= j["low"].to_numpy()) & (y <= j["high"].to_numpy())).mean())
            out["outcomes"] = {
                "with_actual_price": j.height,
                "mdape_pct": round(100 * float(np.median(ape)), 2),
                "within_10_pct": round(100 * float((ape <= 0.10).mean()), 2),
                "range_coverage_pct": round(cov, 2),
                "reference_coverage_pct": ref.get("coverage_pct") if ref else None,
            }
            if j.height >= m["min_feedback"] and cov < m["coverage_alert_below"]:
                out["alerts"].append(
                    f"Range coverage {round(cov, 1)}% (alert below {m['coverage_alert_below']}%)"
                )
    return out


def write_report(path: Path, s: dict[str, Any]) -> None:
    lines = ["# Monitoring", "_Generated by `python -m src.monitor`. Do not edit by hand._", ""]
    if s["requests"] == 0:
        lines.append("No requests logged yet.")
    else:
        lines += ["## Alerts"] + ([f"- {a}" for a in s["alerts"]] or ["- None"]) + [""]
        lines += ["## 1. Pipeline health", md(pl.DataFrame([s["pipeline"]])), ""]
        lines += ["## 2. Drift (PSI: < 0.1 stable, 0.1-0.2 moderate, > 0.2 major)"]
        lines += [
            md(pl.DataFrame(s["drift"]))
            if s["drift"]
            else "Not checked yet: needs the reference (`python -m src.serve build`) "
            "and at least `min_requests` requests.",
            "",
        ]
        lines += ["## 3. Outcomes (listings whose real price came back)"]
        lines += [md(pl.DataFrame([s["outcomes"]])) if "outcomes" in s else "No feedback yet.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- simulate
def simulate(
    cfg: dict[str, Any], n: int, market: str | None, seed: int, form_only: bool = False
) -> None:
    """Replay real listings through the service and send back their real price.

    These listings were in the training data, so accuracy here is optimistic; the point is to
    exercise the pipeline and show the drift checks (e.g. a week of only one market).
    By default the listing's real values are used for every feature; ``form_only`` sends just
    what a host types into the web page (everything else = the market's typical value).
    """
    from src.image_model import unit
    from src.serve import Predictor, append, listing_from_row, log_request

    p = Predictor(cfg)
    df = pl.read_parquet(cfg["paths"]["processed"]).filter(
        pl.col("market").is_in(list(p.form["markets"]))
    )
    if market:
        df = df.filter(pl.col("market") == market)
    df = df.sample(min(n, df.height), seed=seed)
    logs = Path(cfg["serve"]["log_dir"])
    for r in df.iter_rows(named=True):
        x = listing_from_row(r, p.form)
        vec = (
            unit(np.array(r["image_embeddings"], dtype=np.float32))
            if r.get("has_image_emb") == 1
            else None
        )
        out = p.predict(x, photo_vec=vec, known=None if form_only else r)
        log_request(cfg, x, out)
        append(logs / "feedback.jsonl", {"id": out["id"], "actual_price": r["price"]})
    print(f"Replayed {df.height} listings{' from ' + market if market else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: monitoring")
    parser.add_argument("command", nargs="?", choices=["report", "simulate"], default="report")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--market", default=None)
    parser.add_argument("--fresh", action="store_true", help="delete earlier logs first")
    parser.add_argument("--form-only", action="store_true", help="send only the web-form fields")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.command == "simulate":
        if args.fresh:
            for f in Path(cfg["serve"]["log_dir"]).glob("*.jsonl"):
                f.unlink()
        simulate(cfg, args.n, args.market, cfg["seed"], args.form_only)
    s = summarise(cfg)
    path = Path(cfg["paths"]["reports"]) / "monitoring.md"
    with pl.Config(tbl_width_chars=1000, fmt_str_lengths=200):  # keep table headers on one line
        write_report(path, s)
    if s["requests"]:
        print(
            f"Requests {s['requests']}, p95 latency {s['pipeline']['latency_p95_ms']} ms, "
            f"review {s['pipeline']['review_pct']}%"
        )
        if "outcomes" in s:
            o = s["outcomes"]
            print(
                f"Outcomes: MdAPE {o['mdape_pct']}%, range coverage {o['range_coverage_pct']}% "
                f"on {o['with_actual_price']} listings"
            )
    print("Alerts:", *(s["alerts"] or ["none"]), sep="\n  ")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()