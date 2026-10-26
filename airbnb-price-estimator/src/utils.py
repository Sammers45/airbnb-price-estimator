"""Shared helpers used by every phase: config, seeding, logging"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the YAML config into a nested dict"""
    with Path(path).open() as fh:
        return yaml.safe_load(fh)

def set_seed(seed: int) -> None:
    """Seed python and NumPy (and PyTorch once it is installed in a later phase)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch    

        torch.manual_seed(seed)
    except ImportError:
        pass
    logging.getLogger(__name__).info("Seed = %d", seed)

def setup_logging() -> None:
    """One log format for all scripts"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


@contextmanager
def mlflow_run(cfg: dict[str, Any], run_name: str) -> Iterator[Any]:
    """Open an MLFlow run if tracking is enabled in config: otherwise yield None.
    
    Usage:
        with mlflow_run(cfg, "baseline") as mlflow:
            if mlflow is not None:
                mlflow.log_metric("mae", 12.3)
    """
    m = cfg.get("mlflow", {})
    if not m.get("enabled", False):
        yield None
        return
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment"])
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("seed", cfg["seed"])
        yield mlflow


