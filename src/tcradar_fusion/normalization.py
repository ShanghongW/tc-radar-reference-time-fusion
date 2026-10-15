from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import pandas as pd

from .qc import _finite_array, _normalize_zyx


def _partial_stats(task: tuple[str, list[int], dict[str, str], dict[str, Any]]) -> dict[str, float]:
    path, indices, variables, domain = task
    totals = {
        "count_u": 0, "count_v": 0,
        "sum_u": 0.0, "sum_v": 0.0,
        "sumsq_u": 0.0, "sumsq_v": 0.0,
    }
    with nc.Dataset(path) as ds:
        x = _finite_array(ds.variables[variables["x"]][:])
        y = _finite_array(ds.variables[variables["y"]][:])
        z = _finite_array(ds.variables[variables["z"]][:])
        rmax = float(domain["radius_max_km"])
        ix = np.flatnonzero(np.abs(x) <= rmax)
        iy = np.flatnonzero(np.abs(y) <= rmax)
        iz = np.flatnonzero((z >= float(domain["height_min_km"])) & (z <= float(domain["height_max_km"])))
        xx, yy = np.meshgrid(x[ix], y[iy], indexing="xy")
        region = np.broadcast_to((np.hypot(xx, yy) <= rmax)[:, :, None], (len(iy), len(ix), len(iz)))
        uvar = ds.variables[variables["u"]]
        vvar = ds.variables[variables["v"]]
        selected = set(int(index) for index in indices)

        def accumulate(var: nc.Variable, component: str) -> None:
            dims = [str(name).lower() for name in var.dimensions]
            case_axis = next((axis for axis, name in enumerate(dims) if "case" in name), 0)
            y_axis = next(axis for axis, name in enumerate(dims) if "northward" in name or "latitude" in name or name == "y")
            x_axis = next(axis for axis, name in enumerate(dims) if "eastward" in name or "longitude" in name or name == "x")
            z_axis = next(axis for axis, name in enumerate(dims) if "height" in name or name == "z")
            chunking = var.chunking()
            chunk = int(chunking[case_axis]) if isinstance(chunking, list) else len(ds.dimensions["num_cases"])
            for start in range(0, len(ds.dimensions["num_cases"]), chunk):
                stop = min(start + chunk, len(ds.dimensions["num_cases"]))
                offsets = [index - start for index in selected if start <= index < stop]
                if not offsets:
                    continue
                selection: list[Any] = [slice(None)] * len(dims)
                selection[case_axis] = slice(start, stop)
                selection[y_axis] = slice(int(iy[0]), int(iy[-1]) + 1)
                selection[x_axis] = slice(int(ix[0]), int(ix[-1]) + 1)
                selection[z_axis] = slice(int(iz[0]), int(iz[-1]) + 1)
                block = np.ma.asarray(var[tuple(selection)])
                block = np.transpose(block, [case_axis, y_axis, x_axis, z_axis])[offsets]
                data = np.ma.getdata(block)
                valid = ~np.ma.getmaskarray(block) & np.isfinite(data) & region[None, ...]
                values = data[valid].astype(np.float64)
                totals[f"count_{component}"] += int(values.size)
                totals[f"sum_{component}"] += float(values.sum())
                totals[f"sumsq_{component}"] += float(np.square(values).sum())

        accumulate(uvar, "u")
        accumulate(vvar, "v")
    return totals


def compute_training_normalization(swath: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    train = swath[
        (swath["split"] == "train")
        & swath["swath_qc_pass"].astype(bool)
        & swath["parent_mission_fusion_qc_pass"].astype(bool)
    ]
    if train.empty:
        raise RuntimeError("Training split contains no eligible swaths")
    tasks = [
        (str(path), group["swath_case_index"].astype(int).tolist(), cfg["variables"], cfg["domain"])
        for path, group in train.groupby("source_path", sort=True)
    ]
    enabled = bool(cfg["parallel"].get("enabled", True))
    workers = int(cfg["parallel"].get("preprocessing_workers", 1))
    safe_workers = min(workers, int(cfg["parallel"].get("max_concurrent_swath_files", workers)))
    if enabled:
        with ProcessPoolExecutor(max_workers=max(1, min(safe_workers, len(tasks)))) as pool:
            partial = list(pool.map(_partial_stats, tasks))
    else:
        partial = [_partial_stats(task) for task in tasks]
    total = {key: sum(item[key] for item in partial) for key in partial[0]}
    count_u = int(total["count_u"])
    count_v = int(total["count_v"])
    if min(count_u, count_v) < 2:
        raise RuntimeError("Insufficient finite training wind vectors for normalization")
    mean_u = total["sum_u"] / count_u
    mean_v = total["sum_v"] / count_v
    var_u = max(total["sumsq_u"] / count_u - mean_u**2, 1e-8)
    var_v = max(total["sumsq_v"] / count_v - mean_v**2, 1e-8)
    return {
        "u_mean": mean_u,
        "v_mean": mean_v,
        "u_std": float(np.sqrt(var_u)),
        "v_std": float(np.sqrt(var_v)),
        "count_u": count_u,
        "count_v": count_v,
        "determined_from_split": "train",
        "n_training_storms": int(train["storm_key"].nunique()),
        "n_training_missions": int(train["mission_ID"].nunique()),
        "n_training_swaths": int(len(train)),
    }


def write_normalization(stats: dict[str, Any], output_root: str | Path) -> Path:
    path = Path(output_root) / "preprocessing" / "normalization.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_normalization(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
