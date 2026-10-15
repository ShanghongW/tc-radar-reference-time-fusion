from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import netCDF4 as nc
import numpy as np
import pandas as pd
from .geometry import polar_geometry
from .qc import _finite_array, _normalize_zyx, swath_times


@dataclass
class MissionArrays:
    mission_id: str
    storm_key: str
    split: str
    swath_ids: list[str]
    u: np.ndarray
    v: np.ndarray
    times_hours: np.ndarray
    coverages: np.ndarray
    rmw_2km_by_swath: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    reference_index: int
    rmw_2km_km: float


def _load_mission(group: pd.DataFrame, merged_row: pd.Series, cfg: dict[str, Any]) -> MissionArrays:
    group = group.sort_values(["source_file", "swath_case_index"], kind="mergesort").reset_index(drop=True)
    arrays_u: list[np.ndarray] = []
    arrays_v: list[np.ndarray] = []
    midpoints: list[pd.Timestamp] = []
    coverages: list[float] = []
    swath_rmws: list[float] = []
    x = y = z = None
    cached = "cache_path" in group and group["cache_path"].fillna("").map(lambda value: Path(str(value)).exists()).all()
    if cached:
        for _, row in group.iterrows():
            with np.load(row["cache_path"]) as payload:
                x_local = np.asarray(payload["x"], dtype=np.float32)
                y_local = np.asarray(payload["y"], dtype=np.float32)
                z_local = np.asarray(payload["z"], dtype=np.float32)
                if x is None:
                    x, y, z = x_local, y_local, z_local
                elif not (np.array_equal(x, x_local) and np.array_equal(y, y_local) and np.array_equal(z, z_local)):
                    raise ValueError("Cached mission swaths do not share a common recentered grid")
                arrays_u.append(np.asarray(payload["u"], dtype=np.float32))
                arrays_v.append(np.asarray(payload["v"], dtype=np.float32))
            midpoints.append(pd.to_datetime(row["swath_midpoint_time"], errors="coerce"))
            coverages.append(float(row["swath_coverage_2km"]))
            swath_rmws.append(float(row.get("swath_rmw_2km_km", merged_row["rmw_2km_km"])))
    for path, part in ([] if cached else group.groupby("source_path", sort=True)):
        with nc.Dataset(path) as ds:
            x_local = _finite_array(ds.variables[cfg["variables"]["x"]][:])
            y_local = _finite_array(ds.variables[cfg["variables"]["y"]][:])
            z_local = _finite_array(ds.variables[cfg["variables"]["z"]][:])
            if x is None:
                x, y, z = x_local, y_local, z_local
            elif not (np.array_equal(x, x_local) and np.array_equal(y, y_local) and np.array_equal(z, z_local)):
                raise ValueError("Mission swaths do not share a common recentered grid")
            uvar = ds.variables[cfg["variables"]["u"]]
            vvar = ds.variables[cfg["variables"]["v"]]
            for _, row in part.iterrows():
                index = int(row["swath_case_index"])
                arrays_u.append(_normalize_zyx(uvar[index, ...], uvar.dimensions[1:], len(z_local), len(y_local), len(x_local)))
                arrays_v.append(_normalize_zyx(vvar[index, ...], vvar.dimensions[1:], len(z_local), len(y_local), len(x_local)))
                _, _, midpoint, _ = swath_times(ds, index)
                midpoints.append(pd.Timestamp(midpoint) if midpoint is not None else pd.NaT)
                coverages.append(float(row["swath_coverage_2km"]))
                swath_rmws.append(float(row.get("swath_rmw_2km_km", merged_row["rmw_2km_km"])))
    assert x is not None and y is not None and z is not None
                                                                                         
    swath_ids = group["swath_ID"].astype(str).tolist()
    if len(swath_ids) != len(arrays_u):
        raise AssertionError("Swath metadata and arrays are misaligned")
    reference_ids = np.flatnonzero(group["is_primary_reference"].astype(bool).to_numpy())
    if reference_ids.size != 1:
        raise ValueError(f"Mission {merged_row['mission_ID']} must have one primary reference")
    reference_index = int(reference_ids[0])
    ref_time = midpoints[reference_index]
    times = np.asarray([(time - ref_time).total_seconds() / 3600.0 for time in midpoints], dtype=np.float32)
    u = np.stack(arrays_u).astype(np.float32)
    v = np.stack(arrays_v).astype(np.float32)
    radius, _ = polar_geometry(x, y)
    domain = cfg["domain"]
    region = (
        (z[:, None, None] >= float(domain["height_min_km"]))
        & (z[:, None, None] <= float(domain["height_max_km"]))
        & (radius[None, :, :] <= float(domain["radius_max_km"]))
    )
    u[:, ~region] = np.nan
    v[:, ~region] = np.nan
    return MissionArrays(
        mission_id=str(merged_row.name),
        storm_key=str(merged_row["storm_key"]),
        split=str(merged_row["split"]),
        swath_ids=swath_ids,
        u=u,
        v=v,
        times_hours=times,
        coverages=np.asarray(coverages, dtype=np.float32),
        rmw_2km_by_swath=np.asarray(swath_rmws, dtype=np.float32),
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=z.astype(np.float32),
        reference_index=reference_index,
        rmw_2km_km=float(merged_row["rmw_2km_km"]),
    )




def _tables(cfg):
    root = Path(cfg["paths"]["output_root"])
    return pd.read_csv(root / "qc" / "fusion_qc_merged.csv"), pd.read_csv(root / "qc" / "fusion_qc_swath.csv")


def _attach_cache_paths(cfg, swath: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable cache locations without opening NetCDF handles in the parent process."""
    cache_manifest = Path(cfg["paths"]["output_root"]) / "cache" / "swath_cache_manifest.csv"
    if not cache_manifest.exists():
        return swath
    cache = pd.read_csv(cache_manifest)[["swath_ID", "cache_path"]]
    return swath.merge(cache, on="swath_ID", how="left", validate="one_to_one")

