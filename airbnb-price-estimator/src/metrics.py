"""Validate metrics shared by every model.

All metrics ar computed on prices in USD, after converting log predictions back.
"""

from __future__ import annotations

import numpy as np
import polars as pl 

def price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE (USD), median absolute % error, and the share of predictions within 10% / 20%.
    
    Args:
        y_true: Actual prices in USD (> 0).
        y_pred: Predicted prices in USD.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    ape = abs_err / y_true

    return {
        "mae_usd": round(float(abs_err.mean()), 2),
        "mdape_pct": round(float(np.median(ape) * 100), 2),
        "within_10_pct": round(float((ape <= 0.10).mean() * 100), 2),
        "within_20_pct": round(float((ape <= 0.20).mean() * 100), 2),
        "r2_log": round(float(r2(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))), 4),
    }


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination"""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def metrics_by(
        df: pl.DataFrame, group: str, true: str = "price", pred: str = "pred"
) -> pl.DataFrame:
    """`price_metrics` per group, largest groups first."""
    rows = []
    for (key,), part in df.group_by(group, maintain_order=True):
        if part.height < 5:
            continue
        rows.append(
            {
                group: key,
                "rows": part.height,
                **price_metrics(part[true].to_numpy(), part[pred].to_numpy()),
            }
        )
    return pl.DataFrame(rows).sort("rows", descending=True)
