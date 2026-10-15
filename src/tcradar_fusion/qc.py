from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import netCDF4 as nc
import numpy as np
import pandas as pd


def fischer_class_12h(delta_vmax_kt: Any) -> str:
    try:
        value = float(delta_vmax_kt)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not np.isfinite(value):
        return "UNKNOWN"
    if value >= 20.0:
        return "RI"
    if value > 0.0:
        return "SI"
    return "NI"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return str(value).strip()


def _finite_array(value: Any) -> np.ndarray:
    arr = np.ma.asarray(value)
    return np.asarray(np.ma.filled(arr, np.nan), dtype=np.float32)


def _safe_datetime(ds: nc.Dataset, index: int, prefix: str) -> datetime | None:
    minute = f"{prefix}_min"
    if minute not in ds.variables:
        minute = f"{prefix}_minute"
    names = [f"{prefix}_{part}" for part in ("year", "month", "day", "hour")] + [minute]
    try:
        values = [int(ds.variables[name][index]) for name in names]
        return datetime(*values)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def swath_times(ds: nc.Dataset, index: int) -> tuple[datetime | None, datetime | None, datetime | None, str]:
    begin = _safe_datetime(ds, index, "beg_swath")
    end = _safe_datetime(ds, index, "end_swath")
    fix = _safe_datetime(ds, index, "swath")
    if begin is not None and end is not None and end >= begin:
        return begin, end, begin + (end - begin) / 2, "beg_end_midpoint"
    return begin, end, fix, "fix_time_fallback" if fix is not None else "invalid"


def _iso(value: datetime | None) -> str:
    return value.isoformat(timespec="minutes") if value is not None else ""


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def _lag_index(values: np.ndarray, target: float) -> int | None:
    hits = np.flatnonzero(np.isclose(np.asarray(values, dtype=float), target))
    return int(hits[0]) if hits.size else None


def _normalize_zyx(data: np.ndarray, dimensions: Iterable[str], nz: int, ny: int, nx: int) -> np.ndarray:
    arr = _finite_array(data)
    dims = [str(name).lower() for name in dimensions]
    if arr.shape == (ny, nx, nz):
        return np.transpose(arr, (2, 0, 1))
    if arr.shape == (nz, ny, nx):
        return arr
    axes: list[int] = []
    for keys, size in ((["height", "z"], nz), (["northward", "latitude", "y"], ny), (["eastward", "longitude", "x"], nx)):
        candidates = [i for i, (name, n) in enumerate(zip(dims, arr.shape)) if n == size and any(k in name for k in keys)]
        if not candidates:
            candidates = [i for i, n in enumerate(arr.shape) if n == size and i not in axes]
        if not candidates:
            raise ValueError(f"Cannot normalize shape {arr.shape} with dimensions {dimensions}")
        axes.append(candidates[0])
    return np.transpose(arr, tuple(axes))


def _storm_key(tcid: Any, storm_name: Any, mission_time: Any) -> str:
    key = _text(tcid).upper()
    if key:
        return key
    year = str(mission_time)[:4] if str(mission_time) else "0000"
    return f"{_text(storm_name).upper()}_{year}"


def _merged_metadata_for_file(path: str, qc_cfg: dict[str, Any], domain: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with nc.Dataset(path) as ds:
        heights = _finite_array(ds.variables["height"][:])
        lags = _finite_array(ds.variables["ships_lag_times"][:])
        radii = _finite_array(ds.variables["core_radii"][:])
        iz = _nearest_index(heights, domain["reference_height_km"])
        i0, i6, i12 = (_lag_index(lags, lag) for lag in (0.0, 6.0, 12.0))
        if i0 is None or i12 is None:
            raise RuntimeError(f"Required SHIPS lags 0/+12 h missing in {path}")
        ncases = len(ds.dimensions["num_cases"])
        for index in range(ncases):
            mission_id = _text(ds.variables["mission_ID"][index])
            mission_dt = _safe_datetime(ds, index, "merge")
            vmax = _finite_array(ds.variables["vmax_ships"][index, :])
            dtl = _finite_array(ds.variables["dtl_ships"][index, :])
            rmw = float(_finite_array(ds.variables["tc_rmw"][index, iz]))
            core_radius = np.nan
            core_coverage = np.nan
            if np.isfinite(rmw) and rmw > 0:
                ir = _nearest_index(radii, rmw)
                core_radius = float(radii[ir])
                core_coverage = float(_finite_array(ds.variables["core_minimum_quadrant_coverage"][index, ir, iz]))
            tcid = _text(ds.variables["tcid_ships"][index, i0]) if "tcid_ships" in ds.variables else ""
            v0 = float(vmax[i0]) if np.isfinite(vmax[i0]) else np.nan
            v12 = float(vmax[i12]) if np.isfinite(vmax[i12]) else np.nan
            delta = v12 - v0 if np.isfinite(v0) and np.isfinite(v12) else np.nan
            d0 = float(dtl[i0]) if np.isfinite(dtl[i0]) else np.nan
            d6 = float(dtl[i6]) if i6 is not None and np.isfinite(dtl[i6]) else np.nan
            d12 = float(dtl[i12]) if np.isfinite(dtl[i12]) else np.nan
            distances = np.asarray([d0, d6, d12], dtype=float)
            dmin = float(np.nanmin(distances)) if np.any(np.isfinite(distances)) else np.nan
            max_initial_vmax = qc_cfg.get("max_initial_vmax_kt")
            initial_vmax_within_limit = (
                bool(np.isfinite(v0))
                if max_initial_vmax is None
                else bool(np.isfinite(v0) and v0 <= float(max_initial_vmax))
            )
            checks = {
                "fusion_qc_initial_vmax_valid": bool(np.isfinite(v0)),
                "fusion_qc_initial_vmax_le_max": initial_vmax_within_limit,
                "fusion_qc_delta_vmax_12h_valid": bool(np.isfinite(delta)),
                "fusion_qc_distance_to_land": bool(np.all(np.isfinite(distances)) and dmin >= float(qc_cfg["min_distance_to_land_km"])),
                "fusion_qc_rmw_2km_valid": bool(np.isfinite(rmw) and rmw > 0),
                "fusion_qc_core_coverage": bool(np.isfinite(core_coverage) and core_coverage >= float(qc_cfg["min_core_quadrant_coverage"])),
            }
            reasons = [name.removeprefix("fusion_qc_") for name, passed in checks.items() if not passed]
            row = {
                "source_file": Path(path).name,
                "source_path": str(Path(path).resolve()),
                "merged_case_index": index,
                "mission_ID": mission_id,
                "storm_name": _text(ds.variables["storm_name"][index]),
                "mission_time": _iso(mission_dt),
                "tcid": tcid,
                "storm_key": _storm_key(tcid, _text(ds.variables["storm_name"][index]), _iso(mission_dt)),
                "vmax_0h_kt": v0,
                "vmax_12h_kt": v12,
                "delta_vmax_12h_kt": delta,
                "fischer_class_12h": fischer_class_12h(delta),
                "distance_to_land_0h_km": d0,
                "distance_to_land_6h_km": d6,
                "distance_to_land_12h_km": d12,
                "distance_to_land_min_0to12h_km": dmin,
                "rmw_2km_km": rmw,
                "core_coverage_radius_km": core_radius,
                "core_minimum_quadrant_coverage_2km": core_coverage,
                **checks,
                "fusion_merged_pre_qc_pass": bool(all(checks.values())),
                "fusion_merged_pre_qc_fail_reason": ";".join(reasons),
            }
            rows.append(row)
    return rows


def _swath_rows_for_file(path: str, variables: dict[str, str], domain: dict[str, Any], qc_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with nc.Dataset(path) as ds:
        x = _finite_array(ds.variables[variables["x"]][:])
        y = _finite_array(ds.variables[variables["y"]][:])
        z = _finite_array(ds.variables[variables["z"]][:])
        iz2 = _nearest_index(z, domain["reference_height_km"])
        radius_max = float(domain["radius_max_km"])
        ix = np.flatnonzero(np.abs(x) <= radius_max)
        iy = np.flatnonzero(np.abs(y) <= radius_max)
        iz = np.flatnonzero((z >= float(domain["height_min_km"])) & (z <= float(domain["height_max_km"])))
        if not (ix.size and iy.size and iz.size):
            raise ValueError("Configured model domain does not intersect the swath grid")
        x_sub, y_sub = x[ix], y[iy]
        xx, yy = np.meshgrid(x_sub, y_sub, indexing="xy")
        region_yxz = np.broadcast_to((np.hypot(xx, yy) <= radius_max)[:, :, None], (len(iy), len(ix), len(iz)))
        denominator = int(region_yxz.sum())
        uvar = ds.variables.get(variables["u"])
        vvar = ds.variables.get(variables["v"])
        ncases = len(ds.dimensions["num_cases"])
        valid_counts = np.zeros(ncases, dtype=np.int64)
        readable = uvar is not None and vvar is not None
        chunk_size = ncases
        if readable:
            chunking = uvar.chunking()
            if isinstance(chunking, list) and chunking and isinstance(chunking[0], int):
                chunk_size = int(chunking[0])

            def read_valid_block(var: nc.Variable, start: int, stop: int) -> np.ndarray:
                dims = [str(name).lower() for name in var.dimensions]
                selection: list[Any] = [slice(None)] * len(dims)
                case_axis = next((axis for axis, name in enumerate(dims) if "case" in name), 0)
                y_axis = next(axis for axis, name in enumerate(dims) if "northward" in name or "latitude" in name or name == "y")
                x_axis = next(axis for axis, name in enumerate(dims) if "eastward" in name or "longitude" in name or name == "x")
                z_axis = next(axis for axis, name in enumerate(dims) if "height" in name or name == "z")
                selection[case_axis] = slice(start, stop)
                selection[y_axis] = slice(int(iy[0]), int(iy[-1]) + 1)
                selection[x_axis] = slice(int(ix[0]), int(ix[-1]) + 1)
                selection[z_axis] = slice(int(iz[0]), int(iz[-1]) + 1)
                block = np.ma.asarray(var[tuple(selection)])
                data = np.ma.getdata(block)
                valid = ~np.ma.getmaskarray(block) & np.isfinite(data)
                order = [case_axis, y_axis, x_axis, z_axis]
                return np.transpose(valid, order).astype(bool, copy=False)

            try:
                for start in range(0, ncases, chunk_size):
                    stop = min(start + chunk_size, ncases)
                    u_valid = read_valid_block(uvar, start, stop)
                    v_valid = read_valid_block(vvar, start, stop)
                    both = u_valid & v_valid & region_yxz[None, ...]
                    valid_counts[start:stop] = both.sum(axis=(1, 2, 3), dtype=np.int64)
                    del u_valid, v_valid, both
            except Exception:
                readable = False
                valid_counts[:] = 0
        for index in range(ncases):
            mission_id = _text(ds.variables["mission_ID"][index])
            begin, end, midpoint, time_source = swath_times(ds, index)
            coverage = float(_finite_array(ds.variables[variables["swath_coverage"]][index, iz2]))
            swath_rmw = float(_finite_array(ds.variables["tc_rmw"][index, iz2])) if "tc_rmw" in ds.variables else np.nan
            valid_count = int(valid_counts[index])
            valid_ratio = float(valid_count / denominator) if denominator else 0.0
            checks = {
                "swath_qc_time_valid": midpoint is not None,
                "swath_qc_coverage_2km": bool(np.isfinite(coverage) and coverage >= float(qc_cfg["min_swath_coverage_2km"])),
                "swath_qc_uv_readable": bool(readable),
                "swath_qc_uv_valid": bool(readable and valid_count > 0),
            }
            fail = [name.removeprefix("swath_qc_") for name, passed in checks.items() if not passed]
            fingerprint_values = [mission_id, _iso(begin), _iso(end), _iso(midpoint), coverage, valid_count]
            fingerprint = hashlib.sha256("|".join(map(str, fingerprint_values)).encode("utf-8")).hexdigest()
            rows.append({
                "swath_ID": f"{Path(path).name}::case{index:06d}",
                "swath_case_index": index,
                "source_file": Path(path).name,
                "source_path": str(Path(path).resolve()),
                "mission_ID": mission_id,
                "storm_name": _text(ds.variables["storm_name"][index]),
                "swath_start_time": _iso(begin),
                "swath_end_time": _iso(end),
                "swath_midpoint_time": _iso(midpoint),
                "swath_time_source": time_source,
                "swath_coverage_2km": coverage,
                "swath_rmw_2km_km": swath_rmw,
                "valid_uv_count_3d": valid_count,
                "valid_uv_ratio_3d": valid_ratio,
                "valid_uv_method": "direct_finite_u_and_v_native_chunked",
                "uv_native_case_chunk_size": chunk_size,
                "uv_fingerprint": fingerprint,
                **checks,
                "swath_qc_pass_before_duplicate": bool(all(checks.values())),
                "swath_qc_fail_reason_before_duplicate": ";".join(fail),
            })
    return rows


def mission_swath_final_condition(n_good: int, n_bad: int, span_hours: float, cfg: dict[str, Any]) -> tuple[bool, str]:
    reasons: list[str] = []
    if int(n_good) < int(cfg["min_good_swaths"]):
        reasons.append("n_good_below_minimum")
    if bool(cfg.get("require_strict_good_majority", True)) and int(n_good) <= int(n_bad):
        reasons.append("good_not_strict_majority")
    if not np.isfinite(span_hours) or float(span_hours) > float(cfg["max_good_midpoint_span_hours"]):
        reasons.append("good_midpoint_span_exceeds_limit")
    return not reasons, ";".join(reasons)


def choose_primary_reference(rows: pd.DataFrame) -> str:
    eligible = rows[(rows["swath_qc_pass"] == True) & (rows["duplicate_status"] == False)].copy()              
    if eligible.empty:
        return ""
    eligible["_time"] = pd.to_datetime(eligible["swath_midpoint_time"], errors="coerce")
    eligible = eligible.dropna(subset=["_time"]).sort_values(["_time", "swath_case_index", "swath_ID"], kind="mergesort")
    if eligible.empty:
        return ""
    weights = eligible["valid_uv_ratio_3d"].fillna(0.0).clip(lower=0.0).to_numpy(float)
    if weights.sum() <= 0:
        weights = np.ones(len(eligible), dtype=float)
    target_idx = int(np.flatnonzero(np.cumsum(weights) >= 0.5 * weights.sum())[0])
    target = eligible.iloc[target_idx]["_time"]
    eligible["_distance_s"] = (eligible["_time"] - target).abs().dt.total_seconds()
    eligible = eligible.sort_values(
        ["_distance_s", "valid_uv_ratio_3d", "swath_case_index", "swath_ID"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return str(eligible.iloc[0]["swath_ID"])


def mark_duplicate_swaths(swath: pd.DataFrame) -> pd.DataFrame:
    out = swath.copy()
    duplicate = out.duplicated(subset=["uv_fingerprint"], keep="first")
    out["duplicate_status"] = duplicate.astype(bool)
    out["swath_qc_pass"] = out["swath_qc_pass_before_duplicate"].astype(bool) & ~duplicate
    out["swath_qc_fail_reason"] = out["swath_qc_fail_reason_before_duplicate"].fillna("")
    out.loc[duplicate, "swath_qc_fail_reason"] = out.loc[duplicate, "swath_qc_fail_reason"].map(
        lambda value: ";".join(item for item in (str(value), "duplicate_record") if item)
    )
    return out


def _map_parallel(
    function: Any,
    tasks: list[tuple[Any, ...]],
    workers: int,
    enabled: bool,
    force_process: bool = False,
) -> list[Any]:
    if not enabled or (workers <= 1 and not force_process) or (len(tasks) <= 1 and not force_process):
        return [function(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(function, *task) for task in tasks]
        return [future.result() for future in futures]


def build_fusion_qc(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cfg["paths"]
    qc_cfg = cfg["qc"]
    domain = cfg["domain"]
    variables = cfg["variables"]
    parallel = cfg["parallel"]
    enabled = bool(parallel.get("enabled", True))
    workers = int(parallel.get("qc_workers", 1))

    merged_chunks = _map_parallel(
        _merged_metadata_for_file,
        [(path, qc_cfg, domain) for path in paths["merged_files"]],
        workers,
        enabled,
    )
    merged = pd.DataFrame([row for chunk in merged_chunks for row in chunk])

    old = pd.read_csv(paths["qc_all"])
    old = old.rename(columns={"case_index": "merged_case_index"})
    provenance_cols = [col for col in ("source_file", "merged_case_index", "sample_uid", "qc_pass", "qc_fail_reason") if col in old]
    if provenance_cols:
        old_small = old[provenance_cols].rename(columns={"qc_pass": "legacy_qc_pass", "qc_fail_reason": "legacy_qc_fail_reason"})
        merged = merged.merge(old_small, on=["source_file", "merged_case_index"], how="left", validate="one_to_one")

    swath_chunks = _map_parallel(
        _swath_rows_for_file,
        [(path, variables, domain, qc_cfg) for path in paths["swath_files"]],
        min(workers, int(parallel.get("max_concurrent_swath_files", workers))),
        enabled,
        force_process=enabled,
    )
    swath = pd.DataFrame([row for chunk in swath_chunks for row in chunk])
    swath = swath.sort_values(["source_file", "swath_case_index"], kind="mergesort").reset_index(drop=True)
    swath = mark_duplicate_swaths(swath)

    mission_meta = merged.set_index("mission_ID")
    for col in ("storm_key", "storm_name", "vmax_0h_kt", "delta_vmax_12h_kt", "fischer_class_12h"):
        swath[col] = swath["mission_ID"].map(mission_meta[col])

    summary_rows: list[dict[str, Any]] = []
    ref_map: dict[str, str] = {}
    for mission_id, group in swath.groupby("mission_ID", sort=True):
        unique = group[~group["duplicate_status"]]
        good = unique[unique["swath_qc_pass"]]
        bad = unique[~unique["swath_qc_pass"]]
        times = pd.to_datetime(good["swath_midpoint_time"], errors="coerce").dropna()
        span = float((times.max() - times.min()).total_seconds() / 3600.0) if len(times) else np.nan
        swath_condition, swath_reason = mission_swath_final_condition(len(good), len(bad), span, qc_cfg)
        reference = choose_primary_reference(group) if swath_condition else ""
        ref_map[str(mission_id)] = reference
        summary_rows.append({
            "mission_ID": mission_id,
            "n_swaths_raw": int(len(group)),
            "n_swaths_duplicate": int(group["duplicate_status"].sum()),
            "n_swaths_good": int(len(good)),
            "n_swaths_bad": int(len(bad)),
            "good_swath_fraction": float(len(good) / len(unique)) if len(unique) else 0.0,
            "good_swath_time_span_hours": span,
            "fusion_swath_condition_pass": bool(swath_condition),
            "fusion_swath_condition_fail_reason": swath_reason,
            "reference_swath_ID": reference,
        })
    summary = pd.DataFrame(summary_rows)
    merged = merged.merge(summary, on="mission_ID", how="left", validate="one_to_one")
    count_cols = ["n_swaths_raw", "n_swaths_duplicate", "n_swaths_good", "n_swaths_bad"]
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)
    merged["fusion_swath_condition_pass"] = merged["fusion_swath_condition_pass"].fillna(False).astype(bool)
    merged["fusion_qc_pass"] = merged["fusion_merged_pre_qc_pass"].astype(bool) & merged["fusion_swath_condition_pass"]
    merged["fusion_qc_fail_reason"] = merged.apply(
        lambda row: ";".join(
            value for value in (
                str(row.get("fusion_merged_pre_qc_fail_reason", "")),
                str(row.get("fusion_swath_condition_fail_reason", "")),
            ) if value and value != "nan"
        ), axis=1,
    )
    merged["split"] = "excluded"

    final_map = merged.set_index("mission_ID")["fusion_qc_pass"]
    swath["parent_mission_fusion_qc_pass"] = swath["mission_ID"].map(final_map).fillna(False).astype(bool)
    swath["is_primary_reference"] = swath.apply(lambda row: row["swath_ID"] == ref_map.get(str(row["mission_ID"]), ""), axis=1)
    swath["split"] = "excluded"
    merged = merged.sort_values(["source_file", "merged_case_index"], kind="mergesort").reset_index(drop=True)
    return merged, swath


def write_qc_tables(merged: pd.DataFrame, swath: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    out = Path(output_root) / "qc"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "merged": out / "fusion_qc_merged.csv",
        "swath": out / "fusion_qc_swath.csv",
        "merged_pass": out / "fusion_qc_merged_pass.csv",
        "swath_pass": out / "fusion_qc_swath_pass.csv",
    }
    merged.to_csv(paths["merged"], index=False, encoding="utf-8")
    swath.to_csv(paths["swath"], index=False, encoding="utf-8")
    merged[merged["fusion_qc_pass"]].to_csv(paths["merged_pass"], index=False, encoding="utf-8")
    swath[swath["parent_mission_fusion_qc_pass"] & swath["swath_qc_pass"]].to_csv(
        paths["swath_pass"], index=False, encoding="utf-8"
    )
    return paths
