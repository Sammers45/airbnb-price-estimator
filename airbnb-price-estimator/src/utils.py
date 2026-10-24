"""Shared helpers used by every phase: config, seeding, logging"""

from __future__ import annotations

import logging
import os
import random
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