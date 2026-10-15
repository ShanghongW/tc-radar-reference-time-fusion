from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .data_config import load_config as load_parent_config


def load_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the model settings and data-processing configuration."""
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = deepcopy(raw)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    output = (config_path.parent / str(cfg.get("output_root", "outputs"))).resolve()
    cfg["output_root"] = str(output)
    cfg["paths"] = {
        "output_root": str(output),
        "report_root": str(output / "reports"),
        "checkpoint_root": str(output / "checkpoints"),
        "product_root": str(output / "products"),
        "cache_root": str(output / "cache"),
    }
    parent_path = (config_path.parent / str(cfg["parent_config"])).resolve()
    parent = load_parent_config(parent_path)
                                                                             
                                                                 
    cfg["parent_config_path"] = str(parent_path)
    return cfg, parent


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("output_root", "report_root", "checkpoint_root", "product_root", "cache_root"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_reproducible(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

