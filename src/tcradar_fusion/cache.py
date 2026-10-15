from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import pandas as pd

from .qc import _finite_array


def _cache_name(swath_id: str) -> str:
    digest = hashlib.sha256(swath_id.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() else "_" for char in swath_id)[-70:]
    return f"{safe}_{digest}.npz"


def _cache_file_task(task: tuple[str, list[dict[str, Any]], dict[str, str], dict[str, Any], str]) -> list[dict[str, Any]]:
    source_path, records, variables, domain, cache_dir = task
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record_by_case = {int(record["swath_case_index"]): record for record in records}
    selected = set(record_by_case)
    outputs: list[dict[str, Any]] = []
    with nc.Dataset(source_path) as ds:
        x = _finite_array(ds.variables[variables["x"]][:])
        y = _finite_array(ds.variables[variables["y"]][:])
        z = _finite_array(ds.variables[variables["z"]][:])
        rmax = float(domain["radius_max_km"])
        ix = np.flatnonzero(np.abs(x) <= rmax)
        iy = np.flatnonzero(np.abs(y) <= rmax)
        iz = np.flatnonzero((z >= float(domain["height_min_km"])) & (z <= float(domain["height_max_km"])))
        x_sub, y_sub, z_sub = x[ix].astype(np.float32), y[iy].astype(np.float32), z[iz].astype(np.float32)
        xx, yy = np.meshgrid(x_sub, y_sub, indexing="xy")
        region_zyx = np.broadcast_to((np.hypot(xx, yy) <= rmax)[None, :, :], (len(iz), len(iy), len(ix)))
        uvar, vvar = ds.variables[variables["u"]], ds.variables[variables["v"]]
        dims = [str(name).lower() for name in uvar.dimensions]
        case_axis = next((axis for axis, name in enumerate(dims) if "case" in name), 0)
        y_axis = next(axis for axis, name in enumerate(dims) if "northward" in name or "latitude" in name or name == "y")
        x_axis = next(axis for axis, name in enumerate(dims) if "eastward" in name or "longitude" in name or name == "x")
        z_axis = next(axis for axis, name in enumerate(dims) if "height" in name or name == "z")
        chunking = uvar.chunking()
        chunk = int(chunking[case_axis]) if isinstance(chunking, list) else len(ds.dimensions["num_cases"])
        for start in range(0, len(ds.dimensions["num_cases"]), chunk):
            stop = min(start + chunk, len(ds.dimensions["num_cases"]))
            cases = sorted(index for index in selected if start <= index < stop)
            if not cases:
                continue
            selection: list[Any] = [slice(None)] * len(dims)
            selection[case_axis] = slice(start, stop)
            selection[y_axis] = slice(int(iy[0]), int(iy[-1]) + 1)
            selection[x_axis] = slice(int(ix[0]), int(ix[-1]) + 1)
            selection[z_axis] = slice(int(iz[0]), int(iz[-1]) + 1)
            order = [case_axis, z_axis, y_axis, x_axis]
            u_block = np.transpose(np.ma.asarray(uvar[tuple(selection)]), order)
            v_block = np.transpose(np.ma.asarray(vvar[tuple(selection)]), order)
            for case_index in cases:
                offset = case_index - start
                u = np.asarray(np.ma.filled(u_block[offset], np.nan), dtype=np.float32)
                v = np.asarray(np.ma.filled(v_block[offset], np.nan), dtype=np.float32)
                u[~region_zyx] = np.nan
                v[~region_zyx] = np.nan
                swath_id = str(record_by_case[case_index]["swath_ID"])
                path = destination / _cache_name(swath_id)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, u=u, v=v, x=x_sub, y=y_sub, z=z_sub)
                temporary.replace(path)
                outputs.append({
                    "swath_ID": swath_id,
                    "mission_ID": record_by_case[case_index]["mission_ID"],
                    "source_path": source_path,
                    "swath_case_index": case_index,
                    "cache_path": str(path.resolve()),
                    "cache_bytes": path.stat().st_size,
                    "valid_uv_count_cache": int((np.isfinite(u) & np.isfinite(v)).sum()),
                })
            del u_block, v_block
    return outputs


def build_swath_cache(swath: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    selected = swath[swath["parent_mission_fusion_qc_pass"] & swath["swath_qc_pass"]].copy()
    tasks = [
        (str(path), group.to_dict("records"), cfg["variables"], cfg["domain"], cfg["paths"]["cache_dir"])
        for path, group in selected.groupby("source_path", sort=True)
    ]
    workers = min(
        int(cfg["parallel"].get("preprocessing_workers", 1)),
        int(cfg["parallel"].get("max_concurrent_swath_files", 1)),
        len(tasks),
    )
    if bool(cfg["parallel"].get("enabled", True)):
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            chunks = list(pool.map(_cache_file_task, tasks))
    else:
        chunks = [_cache_file_task(task) for task in tasks]
    manifest = pd.DataFrame([row for chunk in chunks for row in chunk]).sort_values(["mission_ID", "swath_case_index"])
    path = Path(cfg["paths"]["output_root"]) / "cache" / "swath_cache_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8")
    if len(manifest) != len(selected) or manifest["swath_ID"].nunique() != len(selected):
        raise AssertionError("Cache manifest is incomplete or contains duplicate swaths")
    return manifest, path

