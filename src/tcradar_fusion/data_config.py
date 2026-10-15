from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml                
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable. "
                "Install PyYAML or keep the configuration in the YAML 1.2 JSON subset."
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise TypeError(f"Configuration root must be a mapping: {path}")
        return loaded


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    cfg = deepcopy(_load_yaml_or_json(config_path))
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    project_root = (config_path.parent / cfg["paths"].get("project_root", "..")).resolve()
    cfg["paths"]["project_root"] = str(project_root)
    for key in (
        "output_root", "cache_dir", "checkpoint_dir", "product_dir",
        "archive_product_dir", "archive_analysis_dir", "time_matching_dir",
        "all_mission_case_diagnostic_dir", "all_mission_case_figure_dir",
        "all_mission_case_diagnostic_500m_dir", "all_mission_case_figure_500m_dir",
        "satellite_validation_dir", "satellite_validation_figure_dir",
    ):
        if key not in cfg["paths"]:
            continue
        value = Path(cfg["paths"][key])
        if not value.is_absolute():
            value = (config_path.parent / value).resolve()
        cfg["paths"][key] = str(value)
    for key in ("qc_all",):
        if key in cfg["paths"]:
            value = Path(cfg["paths"][key]).expanduser()
            cfg["paths"][key] = str(value if value.is_absolute() else (config_path.parent / value).resolve())
    for key in ("merged_files", "swath_files"):
        if key in cfg["paths"]:
            cfg["paths"][key] = [str(Path(v).expanduser() if Path(v).expanduser().is_absolute() else (config_path.parent / v).resolve()) for v in cfg["paths"][key]]
    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in (
        "output_root", "cache_dir", "checkpoint_dir", "product_dir",
        "archive_product_dir", "archive_analysis_dir", "time_matching_dir",
        "all_mission_case_diagnostic_dir", "all_mission_case_figure_dir",
        "all_mission_case_diagnostic_500m_dir", "all_mission_case_figure_500m_dir",
        "satellite_validation_dir", "satellite_validation_figure_dir",
    ):
        if key not in cfg["paths"]:
            continue
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_reproducible(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_seed(base_seed: int, worker_id: int, rank: int = 0) -> int:
    return int((int(base_seed) + 1009 * int(rank) + 9176 * int(worker_id)) % (2**32 - 1))
