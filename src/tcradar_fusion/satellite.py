"""Resolution-aware external satellite validation for one reference-time product.

The module deliberately evaluates a satellite sea-surface wind product as a
near-surface external observation, not as a three-dimensional truth field.
It uses the satellite's native geographic locations, maps both TC-RADAR
products to those locations, and aggregates native polar samples to
area-weighted super-observations at the declared satellite resolution.

No model fitting, calibration, checkpoint selection, or reference selection is
performed here. The archived ML product, primary reference, and validation
calibration are fixed inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from .data_config import load_config


EARTH_RADIUS_KM = 6371.0088
SATELLITE_MIN_SPEED_MS = 0.0
SATELLITE_MAX_SPEED_MS = 100.0


@dataclass(frozen=True)
class SatellitePolarWind:
    """Quality-controlled STAR SAR polar wind-speed product."""

    path: Path
    measurement_time_utc: pd.Timestamp
    latitude: np.ndarray
    longitude: np.ndarray
    wind_speed: np.ndarray
    valid: np.ndarray
    radial_distance_km: np.ndarray
    azimuth_deg_clockwise_from_north: np.ndarray
    center_latitude: float
    center_longitude: float
    source_attributes: dict[str, str]


@dataclass(frozen=True)
class ReferenceFrame:
    """Reference-swath geographic grid and storm-motion conversion."""

    x_km: np.ndarray
    y_km: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    motion_eastward_ms: float
    motion_northward_ms: float
    motion_component_max_abs_deviation_ms: float


@dataclass(frozen=True)
class ProductField:
    """One horizontal product field on an x/y grid."""

    x_km: np.ndarray
    y_km: np.ndarray
    eastward_wind: np.ndarray
    northward_wind: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    provenance: np.ndarray | None = None
    uncertainty_speed: np.ndarray | None = None


def _h5py():
    """Import h5py lazily so the GPU training runtime need not carry it."""

    try:
        import h5py                
    except ImportError as exc:                                                     
        raise RuntimeError(
            "Satellite validation requires h5py. Run this command through the "
            "configured isolated plot runtime, which provides h5py."
        ) from exc
    return h5py


def _deep_update_config(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge nested configuration overrides for a fixed-product comparison."""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_overlay_config_without_torch(path: str | Path) -> dict[str, Any]:
    """Resolve a compact full-model overlay using only the config loader."""

    local = Path(path).expanduser().resolve()
    raw = json.loads(local.read_text(encoding="utf-8"))
    base_path = (local.parent / str(raw.get("base_config", "fusion.yaml"))).resolve()
    cfg = load_config(base_path)
    cfg = _deep_update_config(cfg, raw.get("overrides", {}))
    cfg["full_paths"] = deepcopy(raw.get("full_paths", {}))
    for key in (
        "output_root", "cache_dir", "checkpoint_dir", "product_dir",
        "archive_product_dir", "archive_analysis_dir", "time_matching_dir",
        "satellite_validation_dir", "satellite_validation_figure_dir",
    ):
        if key in cfg["paths"] and not Path(str(cfg["paths"][key])).is_absolute():
            cfg["paths"][key] = str((local.parent / str(cfg["paths"][key])).resolve())
    cfg["_full_config_path"] = str(local)
    cfg["_config_path"] = str(local)
    cfg["_config_dir"] = str(local.parent)
    return cfg


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=str)
    return str(value)


def _as_finite(dataset: Any, selection: Any = Ellipsis) -> np.ndarray:
    """Read a HDF/NetCDF array while honoring common missing-value attributes."""

    values = np.asarray(dataset[selection], dtype=np.float64)
    for key in ("_FillValue", "missing_value"):
        if key not in dataset.attrs:
            continue
        fill = np.asarray(dataset.attrs[key]).reshape(-1)
        for candidate in fill:
            if np.isfinite(candidate):
                values[np.isclose(values, float(candidate), rtol=0.0, atol=0.0)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(_decode(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def read_star_sar_polar(path: str | Path) -> SatellitePolarWind:
    """Read the supplied STAR SAR storm-centred polar HDF5 product.

    The present product contains no per-pixel quality-flag variable. Its QC
    therefore uses finite values plus the physically conservative
    0 < speed <= 100 m s-1 range; zero is not used as a missing-value code.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Satellite HDF5 file is not available: {source}")
    h5py = _h5py()
    with h5py.File(source, "r") as dataset:
        required = ("lat", "lon", "rad", "theta", "wind_speed", "lat_storm_center", "lon_storm_center")
        missing = [name for name in required if name not in dataset]
        if missing:
            raise KeyError(f"STAR SAR HDF5 is missing required fields: {missing}")
        latitude = _as_finite(dataset["lat"])
        longitude = _as_finite(dataset["lon"])
        speed = _as_finite(dataset["wind_speed"])
        radius = _as_finite(dataset["rad"]) / 1000.0
        theta = _as_finite(dataset["theta"])
        if speed.shape != latitude.shape or speed.shape != longitude.shape:
            raise ValueError("Satellite wind, latitude, and longitude arrays must share the same shape")
        if speed.shape != (len(radius), len(theta)):
            raise ValueError(
                "Satellite polar arrays must be ordered as (radial distance, azimuth); "
                f"got speed={speed.shape}, rad={len(radius)}, theta={len(theta)}"
            )
        if "measurementDate" not in dataset.attrs:
            raise KeyError("Satellite product must contain root attribute measurementDate")
        attributes = {str(key): _decode(value) for key, value in dataset.attrs.items()}
        center_latitude = float(_as_finite(dataset["lat_storm_center"]))
        center_longitude = float(_as_finite(dataset["lon_storm_center"]))
    valid = (
        np.isfinite(speed)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & (speed > SATELLITE_MIN_SPEED_MS)
        & (speed <= SATELLITE_MAX_SPEED_MS)
    )
    return SatellitePolarWind(
        path=source,
        measurement_time_utc=_utc_timestamp(attributes["measurementDate"]),
        latitude=latitude,
        longitude=longitude,
        wind_speed=speed,
        valid=valid,
        radial_distance_km=radius,
        azimuth_deg_clockwise_from_north=theta,
        center_latitude=center_latitude,
        center_longitude=center_longitude,
        source_attributes=attributes,
    )


def _nearest_index(axis: np.ndarray, value: float) -> int:
    if not np.isfinite(value):
        raise ValueError("Requested vertical level must be finite")
    return int(np.nanargmin(np.abs(np.asarray(axis, dtype=np.float64) - float(value))))


def _coordinate_grid(dataset: Any, case_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _as_finite(dataset["eastward_distance"])
    y = _as_finite(dataset["northward_distance"])
    latitude = _as_finite(dataset["recentered_latitudes"], case_index)
    longitude = _as_finite(dataset["recentered_longitudes"], case_index)
    expected = (len(y), len(x))
    if latitude.shape != expected or longitude.shape != expected:
        raise ValueError(
            "Recentered geographic coordinate arrays do not match the x/y grid: "
            f"latitude={latitude.shape}, longitude={longitude.shape}, expected={expected}"
        )
    return x, y, latitude, longitude


def _read_case_field(
    dataset: Any,
    variable: str,
    case_index: int,
    level_index: int,
) -> np.ndarray:
    if variable not in dataset:
        raise KeyError(f"Required NetCDF variable is absent: {variable}")
    variable_data = dataset[variable]
    if variable_data.ndim != 4:
        raise ValueError(f"Expected four-dimensional field {variable}; got shape {variable_data.shape}")
    return _as_finite(variable_data, (case_index, slice(None), slice(None), level_index))


def read_reference_frame(
    swath_row: pd.Series,
    target_height_km: float,
) -> ReferenceFrame:
    """Read reference geography and its storm-to-earth-relative conversion."""

    h5py = _h5py()
    source_path = Path(str(swath_row["source_path"])).expanduser()
    case_index = int(swath_row["swath_case_index"])
    with h5py.File(source_path, "r") as dataset:
        z = _as_finite(dataset["height"])
        level_index = _nearest_index(z, target_height_km)
        x, y, latitude, longitude = _coordinate_grid(dataset, case_index)
        storm_u = _read_case_field(dataset, "recentered_eastward_wind", case_index, level_index)
        storm_v = _read_case_field(dataset, "recentered_northward_wind", case_index, level_index)
        earth_u = _read_case_field(dataset, "recentered_earth_relative_eastward_wind", case_index, level_index)
        earth_v = _read_case_field(dataset, "recentered_earth_relative_northward_wind", case_index, level_index)
    delta_u = earth_u - storm_u
    delta_v = earth_v - storm_v
    valid = np.isfinite(delta_u) & np.isfinite(delta_v)
    if not np.any(valid):
        raise ValueError("Reference swath has no finite storm-to-earth-relative conversion values")
    motion_u = float(np.nanmedian(delta_u[valid]))
    motion_v = float(np.nanmedian(delta_v[valid]))
    max_deviation = float(
        max(
            np.nanmax(np.abs(delta_u[valid] - motion_u)),
            np.nanmax(np.abs(delta_v[valid] - motion_v)),
        )
    )
    return ReferenceFrame(
        x_km=x,
        y_km=y,
        latitude=latitude,
        longitude=longitude,
        motion_eastward_ms=motion_u,
        motion_northward_ms=motion_v,
        motion_component_max_abs_deviation_ms=max_deviation,
    )


def read_fused_product(
    path: str | Path,
    reference: ReferenceFrame,
    target_height_km: float,
) -> ProductField:
    """Read archived storm-relative fusion and restore Earth-relative components."""

    product_path = Path(path).expanduser().resolve()
    h5py = _h5py()
    with h5py.File(product_path, "r") as dataset:
        x = _as_finite(dataset["x"])
        y = _as_finite(dataset["y"])
        z = _as_finite(dataset["z"])
        level_index = _nearest_index(z, target_height_km)
        storm_u = _as_finite(dataset["fused_eastward_wind"], (level_index, slice(None), slice(None)))
        storm_v = _as_finite(dataset["fused_northward_wind"], (level_index, slice(None), slice(None)))
        provenance = _as_finite(dataset["fusion_provenance"], (level_index, slice(None), slice(None))).astype(np.int16)
        uncertainty = _as_finite(dataset["fused_uncertainty_speed"], (level_index, slice(None), slice(None)))
    if storm_u.shape != (len(y), len(x)):
        raise ValueError("Fused archive field does not share its declared x/y dimensions")
    ix = np.asarray([int(np.argmin(np.abs(reference.x_km - value))) for value in x])
    iy = np.asarray([int(np.argmin(np.abs(reference.y_km - value))) for value in y])
    if not (
        np.allclose(reference.x_km[ix], x, atol=1.0e-5)
        and np.allclose(reference.y_km[iy], y, atol=1.0e-5)
    ):
        raise ValueError("Archive x/y grid is not a subset of the reference swath geographic grid")
    return ProductField(
        x_km=x,
        y_km=y,
        eastward_wind=storm_u + reference.motion_eastward_ms,
        northward_wind=storm_v + reference.motion_northward_ms,
        latitude=reference.latitude[np.ix_(iy, ix)],
        longitude=reference.longitude[np.ix_(iy, ix)],
        provenance=provenance,
        uncertainty_speed=uncertainty,
    )


def read_official_product(
    merged_row: pd.Series,
    target_height_km: float,
) -> ProductField:
    """Read native Earth-relative official merged winds at the requested level."""

    h5py = _h5py()
    source_path = Path(str(merged_row["source_path"])).expanduser()
    case_index = int(merged_row["merged_case_index"])
    with h5py.File(source_path, "r") as dataset:
        z = _as_finite(dataset["height"])
        level_index = _nearest_index(z, target_height_km)
        x, y, latitude, longitude = _coordinate_grid(dataset, case_index)
        earth_u = _read_case_field(
            dataset,
            "recentered_earth_relative_eastward_wind",
            case_index,
            level_index,
        )
        earth_v = _read_case_field(
            dataset,
            "recentered_earth_relative_northward_wind",
            case_index,
            level_index,
        )
    return ProductField(
        x_km=x,
        y_km=y,
        eastward_wind=earth_u,
        northward_wind=earth_v,
        latitude=latitude,
        longitude=longitude,
    )


class GeographicGridMapper:
    """Map geographic locations back to a supplied recentered x/y grid.

    The inverse is a piecewise-linear interpolation of the stored TC-RADAR
    latitude/longitude coordinate arrays, rather than an assumed map
    projection. This preserves the archive's actual recentering convention.
    """

    def __init__(self, latitude: np.ndarray, longitude: np.ndarray, x_km: np.ndarray, y_km: np.ndarray):
        if latitude.shape != longitude.shape or latitude.shape != (len(y_km), len(x_km)):
            raise ValueError("Latitude/longitude shape does not match the supplied coordinate axes")
        xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
        usable = np.isfinite(latitude) & np.isfinite(longitude)
        points = np.column_stack([longitude[usable], latitude[usable]])
        if len(points) < 4:
            raise ValueError("At least four finite geographic coordinate points are required")
        triangulation = Delaunay(points)
        self._eastward = LinearNDInterpolator(triangulation, xx[usable], fill_value=np.nan)
        self._northward = LinearNDInterpolator(triangulation, yy[usable], fill_value=np.nan)

    def inverse(self, longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.column_stack([np.asarray(longitude).ravel(), np.asarray(latitude).ravel()])
        x = np.asarray(self._eastward(query), dtype=np.float64).reshape(np.asarray(latitude).shape)
        y = np.asarray(self._northward(query), dtype=np.float64).reshape(np.asarray(latitude).shape)
        return x, y


def finite_bilinear_sample(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """Bilinear sample without crossing a field's missing-data boundary."""

    values = np.asarray(field, dtype=np.float64)
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    x_query = np.asarray(x_query, dtype=np.float64)
    y_query = np.asarray(y_query, dtype=np.float64)
    if values.shape != (len(y_axis), len(x_axis)):
        raise ValueError("Field shape must be (len(y_axis), len(x_axis))")
    if not (np.all(np.diff(x_axis) > 0) and np.all(np.diff(y_axis) > 0)):
        raise ValueError("Bilinear sample requires strictly ascending x/y axes")
    result = np.full(x_query.shape, np.nan, dtype=np.float64)
    in_bounds = (
        np.isfinite(x_query)
        & np.isfinite(y_query)
        & (x_query >= x_axis[0])
        & (x_query <= x_axis[-1])
        & (y_query >= y_axis[0])
        & (y_query <= y_axis[-1])
    )
    if not np.any(in_bounds):
        return result
    flat = np.flatnonzero(in_bounds.ravel())
    qx = x_query.ravel()[flat]
    qy = y_query.ravel()[flat]
    ix0 = np.clip(np.searchsorted(x_axis, qx, side="right") - 1, 0, len(x_axis) - 2)
    iy0 = np.clip(np.searchsorted(y_axis, qy, side="right") - 1, 0, len(y_axis) - 2)
    ix1, iy1 = ix0 + 1, iy0 + 1
    tx = (qx - x_axis[ix0]) / (x_axis[ix1] - x_axis[ix0])
    ty = (qy - y_axis[iy0]) / (y_axis[iy1] - y_axis[iy0])
    weights = np.column_stack([
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    ])
    corners = np.column_stack([
        values[iy0, ix0],
        values[iy0, ix1],
        values[iy1, ix0],
        values[iy1, ix1],
    ])
    required = weights > 1.0e-10
    valid = np.all(~required | np.isfinite(corners), axis=1)
    sampled = np.sum(np.where(np.isfinite(corners), corners, 0.0) * weights, axis=1)
    output = result.ravel()
    output[flat[valid]] = sampled[valid]
    return result


def nearest_sample(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour sample for categorical provenance values."""

    field = np.asarray(field)
    x_query = np.asarray(x_query, dtype=np.float64)
    y_query = np.asarray(y_query, dtype=np.float64)
    result = np.full(x_query.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(x_query)
        & np.isfinite(y_query)
        & (x_query >= x_axis[0])
        & (x_query <= x_axis[-1])
        & (y_query >= y_axis[0])
        & (y_query <= y_axis[-1])
    )
    if not np.any(valid):
        return result
    positions = np.flatnonzero(valid.ravel())
    qx, qy = x_query.ravel()[positions], y_query.ravel()[positions]
    ix = np.searchsorted(x_axis, qx, side="left")
    iy = np.searchsorted(y_axis, qy, side="left")
    ix = np.clip(ix, 0, len(x_axis) - 1)
    iy = np.clip(iy, 0, len(y_axis) - 1)
    previous_x = np.clip(ix - 1, 0, len(x_axis) - 1)
    previous_y = np.clip(iy - 1, 0, len(y_axis) - 1)
    ix = np.where(np.abs(x_axis[previous_x] - qx) <= np.abs(x_axis[ix] - qx), previous_x, ix)
    iy = np.where(np.abs(y_axis[previous_y] - qy) <= np.abs(y_axis[iy] - qy), previous_y, iy)
    result.ravel()[positions] = field[iy, ix]
    return result


def _ring_cell_area_weights(radius_km: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_km, dtype=np.float64)
    theta = np.asarray(theta_deg, dtype=np.float64)
    if len(radius) < 2 or len(theta) < 2:
        raise ValueError("Satellite polar grid must have at least two radial and angular positions")
    radial_edges = np.empty(len(radius) + 1, dtype=np.float64)
    radial_edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
    radial_edges[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
    radial_edges[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])
    theta_step_rad = math.radians(float(np.nanmedian(np.diff(theta))))
    ring_area = 0.5 * (np.square(radial_edges[1:]) - np.square(radial_edges[:-1])) * theta_step_rad
    return np.broadcast_to(ring_area[:, None], (len(radius), len(theta))).copy()


def build_satellite_superobservations(
    satellite: SatellitePolarWind,
    resolution_km: float,
    radius_limit_km: float,
    min_cell_coverage: float,
) -> pd.DataFrame:
    """Aggregate raw polar nodes to equal-resolution Cartesian super-observations.

    Each native polar node receives its annular-sector area weight. A
    super-observation is retained only when valid satellite wind covers the
    configured fraction of the Cartesian cell. The returned rows retain
    original geographic positions for later model co-location.
    """

    resolution_km = float(resolution_km)
    radius_limit_km = float(radius_limit_km)
    min_cell_coverage = float(min_cell_coverage)
    if resolution_km <= 0.0 or radius_limit_km <= 0.0:
        raise ValueError("Satellite resolution and radius limit must be positive")
    if not 0.0 < min_cell_coverage <= 1.0:
        raise ValueError("min_cell_coverage must be in (0, 1]")
    radial, theta = np.meshgrid(
        satellite.radial_distance_km,
        np.deg2rad(satellite.azimuth_deg_clockwise_from_north),
        indexing="ij",
    )
    x = radial * np.sin(theta)
    y = radial * np.cos(theta)
    area = _ring_cell_area_weights(
        satellite.radial_distance_km,
        satellite.azimuth_deg_clockwise_from_north,
    )
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (radial <= radius_limit_km + 1.0e-8)
        & (np.abs(x) <= radius_limit_km + 1.0e-8)
        & (np.abs(y) <= radius_limit_km + 1.0e-8)
    )
    ncell = int(math.ceil(2.0 * radius_limit_km / resolution_km))
    x_flat, y_flat = x.ravel(), y.ravel()
    ix = np.floor((x_flat + radius_limit_km) / resolution_km).astype(int)
    iy = np.floor((y_flat + radius_limit_km) / resolution_km).astype(int)
    ix = np.clip(ix, 0, ncell - 1)
    iy = np.clip(iy, 0, ncell - 1)
    cell = iy * ncell + ix
    inside_flat = inside.ravel()
    weights = area.ravel()
    total_area = np.bincount(
        cell[inside_flat],
        weights=weights[inside_flat],
        minlength=ncell * ncell,
    )
    valid_flat = satellite.valid.ravel() & inside_flat
    satellite_area = np.bincount(
        cell[valid_flat],
        weights=weights[valid_flat],
        minlength=ncell * ncell,
    )
    coverage = np.divide(
        satellite_area,
        total_area,
        out=np.zeros_like(satellite_area),
        where=total_area > 0.0,
    )
    keep = valid_flat & (coverage[cell] >= min_cell_coverage)
    selected = np.flatnonzero(keep)
    if selected.size == 0:
        return pd.DataFrame(
            columns=[
                "supercell_id", "supercell_x_km", "supercell_y_km",
                "satellite_x_km", "satellite_y_km", "latitude", "longitude",
                "satellite_speed_ms", "area_weight_km2", "satellite_cell_coverage",
                "satellite_valid_area_km2", "satellite_total_area_km2",
            ]
        )
    selected_cell = cell[selected]
    selected_ix, selected_iy = ix[selected], iy[selected]
    return pd.DataFrame(
        {
            "supercell_id": selected_cell.astype(np.int64),
            "supercell_x_km": -radius_limit_km + (selected_ix + 0.5) * resolution_km,
            "supercell_y_km": -radius_limit_km + (selected_iy + 0.5) * resolution_km,
            "satellite_x_km": x_flat[selected],
            "satellite_y_km": y_flat[selected],
            "latitude": satellite.latitude.ravel()[selected],
            "longitude": satellite.longitude.ravel()[selected],
            "satellite_speed_ms": satellite.wind_speed.ravel()[selected],
            "area_weight_km2": weights[selected],
            "satellite_cell_coverage": coverage[selected_cell],
            "satellite_valid_area_km2": satellite_area[selected_cell],
            "satellite_total_area_km2": total_area[selected_cell],
        }
    )


def colocate_satellite_nodes(
    nodes: pd.DataFrame,
    fused: ProductField,
    official: ProductField,
) -> pd.DataFrame:
    """Sample both products at each retained native satellite observation."""

    if nodes.empty:
        return nodes.copy()
    fused_mapper = GeographicGridMapper(fused.latitude, fused.longitude, fused.x_km, fused.y_km)
    official_mapper = GeographicGridMapper(official.latitude, official.longitude, official.x_km, official.y_km)
    latitude = nodes["latitude"].to_numpy(dtype=np.float64)
    longitude = nodes["longitude"].to_numpy(dtype=np.float64)
    fused_x, fused_y = fused_mapper.inverse(longitude, latitude)
    official_x, official_y = official_mapper.inverse(longitude, latitude)
    fused_u = finite_bilinear_sample(fused.eastward_wind, fused.x_km, fused.y_km, fused_x, fused_y)
    fused_v = finite_bilinear_sample(fused.northward_wind, fused.x_km, fused.y_km, fused_x, fused_y)
    official_u = finite_bilinear_sample(official.eastward_wind, official.x_km, official.y_km, official_x, official_y)
    official_v = finite_bilinear_sample(official.northward_wind, official.x_km, official.y_km, official_x, official_y)
    result = nodes.copy()
    result["fused_x_km"] = fused_x
    result["fused_y_km"] = fused_y
    result["official_x_km"] = official_x
    result["official_y_km"] = official_y
    result["fused_speed_ms"] = np.hypot(fused_u, fused_v)
    result["official_speed_ms"] = np.hypot(official_u, official_v)
    if fused.provenance is not None:
        result["fusion_provenance"] = nearest_sample(
            fused.provenance,
            fused.x_km,
            fused.y_km,
            fused_x,
            fused_y,
        )
    else:
        result["fusion_provenance"] = np.nan
    if fused.uncertainty_speed is not None:
        result["fused_uncertainty_speed_ms"] = finite_bilinear_sample(
            fused.uncertainty_speed,
            fused.x_km,
            fused.y_km,
            fused_x,
            fused_y,
        )
    else:
        result["fused_uncertainty_speed_ms"] = np.nan
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return float("nan")
    return float(np.sum(values[usable] * weights[usable]) / np.sum(weights[usable]))


def aggregate_common_superobservations(
    nodes: pd.DataFrame,
    min_model_coverage: float,
) -> pd.DataFrame:
    """Make common-support, area-weighted satellite/model cells."""

    if nodes.empty:
        return pd.DataFrame()
    min_model_coverage = float(min_model_coverage)
    if not 0.0 < min_model_coverage <= 1.0:
        raise ValueError("min_model_coverage must be in (0, 1]")
    common = (
        np.isfinite(nodes["satellite_speed_ms"].to_numpy())
        & np.isfinite(nodes["fused_speed_ms"].to_numpy())
        & np.isfinite(nodes["official_speed_ms"].to_numpy())
    )
    working = nodes.loc[common].copy()
    records: list[dict[str, float | int]] = []
    for supercell_id, group in working.groupby("supercell_id", sort=True):
        weights = group["area_weight_km2"].to_numpy(dtype=np.float64)
        common_area = float(np.sum(weights))
        satellite_area = float(group["satellite_valid_area_km2"].iloc[0])
        if satellite_area <= 0.0 or common_area / satellite_area < min_model_coverage:
            continue
        sat = group["satellite_speed_ms"].to_numpy(dtype=np.float64)
        fused = group["fused_speed_ms"].to_numpy(dtype=np.float64)
        official = group["official_speed_ms"].to_numpy(dtype=np.float64)
        provenance = group["fusion_provenance"].to_numpy(dtype=np.float64)
        records.append(
            {
                "supercell_id": int(supercell_id),
                "x_km": float(group["supercell_x_km"].iloc[0]),
                "y_km": float(group["supercell_y_km"].iloc[0]),
                "radius_km": float(np.hypot(group["supercell_x_km"].iloc[0], group["supercell_y_km"].iloc[0])),
                "latitude": _weighted_mean(group["latitude"].to_numpy(dtype=np.float64), weights),
                "longitude": _weighted_mean(group["longitude"].to_numpy(dtype=np.float64), weights),
                "satellite_speed_ms": _weighted_mean(sat, weights),
                "fused_speed_ms": _weighted_mean(fused, weights),
                "official_speed_ms": _weighted_mean(official, weights),
                "fused_uncertainty_speed_ms": _weighted_mean(
                    group["fused_uncertainty_speed_ms"].to_numpy(dtype=np.float64), weights
                ),
                "area_weight_km2": common_area,
                "satellite_cell_coverage": float(group["satellite_cell_coverage"].iloc[0]),
                "model_common_coverage": common_area / satellite_area,
                "n_native_satellite_nodes": int(len(group)),
                "reference_fraction": _weighted_mean((provenance == 1).astype(float), weights),
                "one_source_fraction": _weighted_mean((provenance == 2).astype(float), weights),
                "multiple_source_fraction": _weighted_mean((provenance == 3).astype(float), weights),
                "candidate_fraction": _weighted_mean(np.isin(provenance, [2, 3]).astype(float), weights),
            }
        )
    cells = pd.DataFrame(records)
    if cells.empty:
        return cells
    cells["fused_error_ms"] = cells["fused_speed_ms"] - cells["satellite_speed_ms"]
    cells["official_error_ms"] = cells["official_speed_ms"] - cells["satellite_speed_ms"]
    cells["absolute_error_reduction_ms"] = (
        np.abs(cells["official_error_ms"]) - np.abs(cells["fused_error_ms"])
    )
    block_size_km = 15.0
    bx = np.floor((cells["x_km"] + 150.0) / block_size_km).astype(int)
    by = np.floor((cells["y_km"] + 150.0) / block_size_km).astype(int)
    cells["spatial_block"] = bx.astype(str) + "_" + by.astype(str)
    return cells


def _weighted_correlation(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(first) & np.isfinite(second) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(usable) < 3:
        return float("nan")
    x, y, w = first[usable], second[usable], weights[usable]
    mx, my = _weighted_mean(x, w), _weighted_mean(y, w)
    vx = _weighted_mean(np.square(x - mx), w)
    vy = _weighted_mean(np.square(y - my), w)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = _weighted_mean((x - mx) * (y - my), w)
    return float(cov / math.sqrt(vx * vy))


def _radial_structure_metrics(cells: pd.DataFrame, bin_width_km: float = 15.0) -> dict[str, float | int]:
    """Compare radial speed structure without treating satellite pixels as iid.

    The profile is formed from the area-weighted Cartesian super-observations
    already used for the external comparison.  Rings with fewer than three
    cells are not used.  Peak statistics refer to this observed common-support
    radial profile, not to a vortex-wide or best-track RMW estimate.
    """

    empty: dict[str, float | int] = {
        "radial_profile_bin_width_km": float(bin_width_km),
        "n_radial_profile_bins": 0,
        "fused_radial_profile_rmse_ms": float("nan"),
        "official_radial_profile_rmse_ms": float("nan"),
        "fused_radial_profile_correlation": float("nan"),
        "official_radial_profile_correlation": float("nan"),
        "satellite_profile_peak_speed_ms": float("nan"),
        "fused_profile_peak_speed_ms": float("nan"),
        "official_profile_peak_speed_ms": float("nan"),
        "fused_profile_peak_speed_abs_error_ms": float("nan"),
        "official_profile_peak_speed_abs_error_ms": float("nan"),
        "satellite_profile_peak_radius_km": float("nan"),
        "fused_profile_peak_radius_km": float("nan"),
        "official_profile_peak_radius_km": float("nan"),
        "fused_profile_peak_radius_abs_error_km": float("nan"),
        "official_profile_peak_radius_abs_error_km": float("nan"),
    }
    if cells.empty:
        return empty
    radius = cells["radius_km"].to_numpy(dtype=np.float64)
    max_radius = float(np.nanmax(radius)) if np.any(np.isfinite(radius)) else 0.0
    if max_radius <= 0.0:
        return empty
    edges = np.arange(0.0, max_radius + bin_width_km * 1.01, bin_width_km)
    records: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        subset = cells[(cells["radius_km"] >= low) & (cells["radius_km"] < high)]
        if len(subset) < 3:
            continue
        weights = subset["area_weight_km2"].to_numpy(dtype=np.float64)
        sat = _weighted_mean(subset["satellite_speed_ms"].to_numpy(dtype=np.float64), weights)
        fused = _weighted_mean(subset["fused_speed_ms"].to_numpy(dtype=np.float64), weights)
        official = _weighted_mean(subset["official_speed_ms"].to_numpy(dtype=np.float64), weights)
        if not np.all(np.isfinite([sat, fused, official])):
            continue
        records.append(
            {
                "radius_km": 0.5 * (low + high),
                "area_weight_km2": float(np.sum(weights)),
                "satellite_speed_ms": sat,
                "fused_speed_ms": fused,
                "official_speed_ms": official,
            }
        )
    if not records:
        return empty
    profile = pd.DataFrame(records)
    weights = profile["area_weight_km2"].to_numpy(dtype=np.float64)
    sat = profile["satellite_speed_ms"].to_numpy(dtype=np.float64)
    fused = profile["fused_speed_ms"].to_numpy(dtype=np.float64)
    official = profile["official_speed_ms"].to_numpy(dtype=np.float64)
    satellite_peak = int(np.nanargmax(sat))
    fused_peak = int(np.nanargmax(fused))
    official_peak = int(np.nanargmax(official))
    return {
        "radial_profile_bin_width_km": float(bin_width_km),
        "n_radial_profile_bins": int(len(profile)),
        "fused_radial_profile_rmse_ms": float(math.sqrt(_weighted_mean(np.square(fused - sat), weights))),
        "official_radial_profile_rmse_ms": float(math.sqrt(_weighted_mean(np.square(official - sat), weights))),
        "fused_radial_profile_correlation": _weighted_correlation(fused, sat, weights),
        "official_radial_profile_correlation": _weighted_correlation(official, sat, weights),
        "satellite_profile_peak_speed_ms": float(sat[satellite_peak]),
        "fused_profile_peak_speed_ms": float(fused[fused_peak]),
        "official_profile_peak_speed_ms": float(official[official_peak]),
        "fused_profile_peak_speed_abs_error_ms": float(abs(fused[fused_peak] - sat[satellite_peak])),
        "official_profile_peak_speed_abs_error_ms": float(abs(official[official_peak] - sat[satellite_peak])),
        "satellite_profile_peak_radius_km": float(profile.iloc[satellite_peak]["radius_km"]),
        "fused_profile_peak_radius_km": float(profile.iloc[fused_peak]["radius_km"]),
        "official_profile_peak_radius_km": float(profile.iloc[official_peak]["radius_km"]),
        "fused_profile_peak_radius_abs_error_km": float(
            abs(profile.iloc[fused_peak]["radius_km"] - profile.iloc[satellite_peak]["radius_km"])
        ),
        "official_profile_peak_radius_abs_error_km": float(
            abs(profile.iloc[official_peak]["radius_km"] - profile.iloc[satellite_peak]["radius_km"])
        ),
    }


def _block_bootstrap_delta_mae(
    cells: pd.DataFrame,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Spatial-block bootstrap CI for official-minus-fusion MAE reduction."""

    if cells.empty or cells["spatial_block"].nunique() < 2 or int(resamples) <= 1:
        return float("nan"), float("nan")
    block = cells.groupby("spatial_block", sort=True)
    block_weights = block.apply(lambda frame: float(np.sum(frame["area_weight_km2"]))).to_numpy(dtype=np.float64)
    block_numerators = block.apply(
        lambda frame: float(np.sum(frame["area_weight_km2"] * frame["absolute_error_reduction_ms"]))
    ).to_numpy(dtype=np.float64)
    if len(block_weights) < 2 or not np.any(block_weights > 0.0):
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(block_weights), size=(int(resamples), len(block_weights)))
    denominator = block_weights[choices].sum(axis=1)
    numerator = block_numerators[choices].sum(axis=1)
    samples = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)
    return float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))


def summarize_external_metrics(
    cells: pd.DataFrame,
    scope: str,
    seed: int,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Summarize wind-speed error at one independent-super-observation scope."""

    if cells.empty:
        return {
            "scope": scope,
            "n_superobservations": 0,
            "n_spatial_blocks": 0,
        }
    sat = cells["satellite_speed_ms"].to_numpy(dtype=np.float64)
    fused = cells["fused_speed_ms"].to_numpy(dtype=np.float64)
    official = cells["official_speed_ms"].to_numpy(dtype=np.float64)
    weights = cells["area_weight_km2"].to_numpy(dtype=np.float64)
    fused_error = fused - sat
    official_error = official - sat
    mae_fused = _weighted_mean(np.abs(fused_error), weights)
    mae_official = _weighted_mean(np.abs(official_error), weights)
    rmse_fused = math.sqrt(_weighted_mean(np.square(fused_error), weights))
    rmse_official = math.sqrt(_weighted_mean(np.square(official_error), weights))
    ci_low, ci_high = _block_bootstrap_delta_mae(cells, bootstrap_resamples, seed)
    return {
        "scope": scope,
        "n_superobservations": int(len(cells)),
        "n_spatial_blocks": int(cells["spatial_block"].nunique()),
        "area_km2": float(np.sum(weights)),
        "satellite_mean_speed_ms": _weighted_mean(sat, weights),
        "fused_bias_ms": _weighted_mean(fused_error, weights),
        "official_bias_ms": _weighted_mean(official_error, weights),
        "fused_mae_ms": mae_fused,
        "official_mae_ms": mae_official,
        "fused_rmse_ms": rmse_fused,
        "official_rmse_ms": rmse_official,
        "fused_correlation": _weighted_correlation(fused, sat, weights),
        "official_correlation": _weighted_correlation(official, sat, weights),
        "mae_reduction_official_minus_fused_ms": mae_official - mae_fused,
        "rmse_reduction_official_minus_fused_ms": rmse_official - rmse_fused,
        "fraction_cells_fusion_lower_absolute_error": _weighted_mean(
            (np.abs(fused_error) < np.abs(official_error)).astype(float),
            weights,
        ),
        "mae_reduction_block_bootstrap_ci95_low_ms": ci_low,
        "mae_reduction_block_bootstrap_ci95_high_ms": ci_high,
        **_radial_structure_metrics(cells),
    }


def _scope_tables(cells: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("all_common_support", cells),
        ("reference_dominant", cells[cells["reference_fraction"] >= 0.5]),
        ("candidate_dominant", cells[cells["candidate_fraction"] >= 0.5]),
        ("one_source_dominant", cells[cells["one_source_fraction"] >= 0.5]),
        ("multiple_source_dominant", cells[cells["multiple_source_fraction"] >= 0.5]),
    ]


def _nice_upper(values: np.ndarray) -> float:
    usable = np.asarray(values, dtype=np.float64)
    usable = usable[np.isfinite(usable)]
    if usable.size == 0:
        return 10.0
    high = float(max(np.nanpercentile(usable, 99.0), np.nanmax(usable)))
    if high <= 30.0:
        step = 5.0
    elif high <= 80.0:
        step = 10.0
    else:
        step = 20.0
    return max(step, math.ceil(high / step) * step)


def _nice_symmetric_limit(values: np.ndarray) -> float:
    usable = np.abs(np.asarray(values, dtype=np.float64))
    usable = usable[np.isfinite(usable)]
    if usable.size == 0:
        return 5.0
    high = float(max(np.nanpercentile(usable, 99.0), np.nanmax(usable)))
    step = 2.0 if high <= 12.0 else 5.0 if high <= 35.0 else 10.0
    return max(step, math.ceil(high / step) * step)


def _style_axes(axis: Any) -> None:
    from matplotlib.ticker import AutoMinorLocator

    for spine in ("left", "bottom", "top", "right"):
        axis.spines[spine].set_visible(True)
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.7,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    axis.tick_params(axis="both", which="minor", direction="out", length=1.8, width=0.5, top=True, right=True)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=10,
    )


def _draw_rings(axis: Any, radius_limit_km: float) -> None:
    from matplotlib.patches import Circle

    for radius in (50.0, 100.0, 150.0):
        if radius <= radius_limit_km + 1.0e-6:
            axis.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor="0.35", linewidth=0.35, alpha=0.45))


def _field_scatter(axis: Any, cells: pd.DataFrame, column: str, cmap: str, vmin: float, vmax: float):
    return axis.scatter(
        cells["x_km"],
        cells["y_km"],
        c=cells[column],
        s=15.0,
        marker="s",
        linewidths=0.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def make_satellite_validation_figures(
    cells: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
    mission_id: str,
    storm_name: str,
    satellite_time: pd.Timestamp,
    reference_time: pd.Timestamp,
    official_time: pd.Timestamp,
    resolution_km: float,
    radius_limit_km: float,
    metric_scope: str = "all_common_support",
    figure_tag: str = "",
    scope_label: str = "all common-support cells",
) -> dict[str, Path]:
    """Make main-field and radial/scatter satellite validation figures."""

    import matplotlib.pyplot as plt

    style = Path(__file__).with_name("publication.mplstyle")
    if style.is_file():
        plt.style.use(style)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cells.empty:
        raise ValueError("Cannot plot an empty satellite/model common-support sample")
    speed_max = _nice_upper(
        np.concatenate(
            [
                cells["satellite_speed_ms"].to_numpy(),
                cells["fused_speed_ms"].to_numpy(),
                cells["official_speed_ms"].to_numpy(),
            ]
        )
    )
    error_limit = _nice_symmetric_limit(
        np.concatenate([cells["fused_error_ms"].to_numpy(), cells["official_error_ms"].to_numpy()])
    )
    reduction_limit = _nice_symmetric_limit(cells["absolute_error_reduction_ms"].to_numpy())
    bottom_limit = max(error_limit, reduction_limit)
    scoped_metrics = metrics[metrics["scope"] == metric_scope]
    if scoped_metrics.empty:
        raise ValueError(f"Missing metrics for plotted scope {metric_scope}")
    all_metrics = scoped_metrics.iloc[0]
    figure = plt.figure(figsize=(7.15, 5.45))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.065),
        left=0.075,
        right=0.965,
        bottom=0.125,
        top=0.975,
        wspace=0.34,
        hspace=0.32,
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, column]) for column in range(3)],
            [figure.add_subplot(grid[1, column]) for column in range(3)],
        ]
    )
    speed_cax = figure.add_subplot(grid[0, 3])
    difference_cax = figure.add_subplot(grid[1, 3])
    top_columns = [
        ("satellite_speed_ms", "(a) Satellite 10 m wind speed"),
        ("fused_speed_ms", "(b) ML fusion at 0.5 km"),
        ("official_speed_ms", "(c) Official merge at 0.5 km"),
    ]
    top_artists = []
    for axis, (column, label) in zip(axes[0], top_columns):
        artist = _field_scatter(axis, cells, column, "YlOrRd", 0.0, speed_max)
        top_artists.append(artist)
        _draw_rings(axis, radius_limit_km)
        _style_axes(axis)
        _panel_label(axis, label)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-radius_limit_km, radius_limit_km)
        axis.set_ylim(-radius_limit_km, radius_limit_km)
    difference_columns = [
        ("fused_error_ms", "(d) ML fusion minus satellite"),
        ("official_error_ms", "(e) Official merge minus satellite"),
        ("absolute_error_reduction_ms", "(f) |official error| minus |fusion error|"),
    ]
    bottom_artists = []
    for axis, (column, label) in zip(axes[1], difference_columns):
        artist = _field_scatter(axis, cells, column, "RdBu_r", -bottom_limit, bottom_limit)
        bottom_artists.append(artist)
        _draw_rings(axis, radius_limit_km)
        _style_axes(axis)
        _panel_label(axis, label)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-radius_limit_km, radius_limit_km)
        axis.set_ylim(-radius_limit_km, radius_limit_km)
    for axis in axes[:, 0]:
        axis.set_ylabel("Northward distance (km)")
    for axis in axes[1]:
        axis.set_xlabel("Eastward distance (km)")
    speed_bar = figure.colorbar(top_artists[-1], cax=speed_cax)
    speed_bar.set_label("Earth-relative wind speed (m s$^{-1}$)")
    error_bar = figure.colorbar(bottom_artists[-1], cax=difference_cax)
    error_bar.set_label("Difference / error reduction (m s$^{-1}$)")
    reference_offset = (satellite_time - reference_time).total_seconds() / 60.0
    official_offset = (official_time - satellite_time).total_seconds() / 60.0
    footnote = textwrap.fill(
        (
            f"{storm_name} {mission_id}; {scope_label}; {resolution_km:g}-km area-weighted super-observations "
            f"(N={int(all_metrics['n_superobservations'])}). Satellite {satellite_time:%Y-%m-%d %H:%M:%SZ}; "
            f"reference {reference_time:%H:%MZ} ({reference_offset:+.1f} min); "
            f"official merge {official_time:%H:%MZ} ({official_offset:+.1f} min). "
            f"All comparisons are satellite-10-m versus TC-RADAR-0.5-km near-surface diagnostics."
        ),
        width=150,
    )
    figure.text(
        0.5,
        0.006,
        footnote,
        ha="center",
        va="bottom",
        fontsize=6.7,
    )
    field_pdf = output / f"fig_satellite_validation_{mission_id}{figure_tag}.pdf"
    field_png = output / f"fig_satellite_validation_{mission_id}{figure_tag}.png"
    figure.savefig(field_pdf, bbox_inches="tight")
    figure.savefig(field_png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(6.90, 4.75), constrained_layout=True)
    radial_edges = np.arange(0.0, radius_limit_km + 25.0, 25.0)
    radial_center = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    radial_records: list[dict[str, float]] = []
    for low, high, center in zip(radial_edges[:-1], radial_edges[1:], radial_center):
        subset = cells[(cells["radius_km"] >= low) & (cells["radius_km"] < high)]
        if subset.empty:
            continue
        weight = subset["area_weight_km2"].to_numpy(dtype=np.float64)
        radial_records.append(
            {
                "radius": center,
                "satellite": _weighted_mean(subset["satellite_speed_ms"].to_numpy(), weight),
                "fusion": _weighted_mean(subset["fused_speed_ms"].to_numpy(), weight),
                "official": _weighted_mean(subset["official_speed_ms"].to_numpy(), weight),
                "fusion_mae": _weighted_mean(np.abs(subset["fused_error_ms"].to_numpy()), weight),
                "official_mae": _weighted_mean(np.abs(subset["official_error_ms"].to_numpy()), weight),
                "reduction": _weighted_mean(subset["absolute_error_reduction_ms"].to_numpy(), weight),
            }
        )
    radial = pd.DataFrame(radial_records)
    axis = axes[0, 0]
    if not radial.empty:
        axis.plot(radial["radius"], radial["satellite"], color="black", linewidth=1.25, label="Satellite")
        axis.plot(radial["radius"], radial["fusion"], color="tab:red", linewidth=1.25, label="ML fusion")
        axis.plot(radial["radius"], radial["official"], color="tab:blue", linewidth=1.25, label="Official merge")
    axis.set(xlabel="Radius from satellite center (km)", ylabel="Wind speed (m s$^{-1}$)")
    axis.legend(frameon=False, fontsize=7, loc="best")
    _style_axes(axis)
    _panel_label(axis, "(a) Radial mean wind speed")
    axis = axes[0, 1]
    axis.scatter(cells["satellite_speed_ms"], cells["fused_speed_ms"], s=8, color="tab:red", alpha=0.42, label="ML fusion", rasterized=True)
    axis.scatter(cells["satellite_speed_ms"], cells["official_speed_ms"], s=8, color="tab:blue", alpha=0.35, label="Official merge", rasterized=True)
    line_max = _nice_upper(
        np.concatenate(
            [
                cells["satellite_speed_ms"].to_numpy(),
                cells["fused_speed_ms"].to_numpy(),
                cells["official_speed_ms"].to_numpy(),
            ]
        )
    )
    axis.plot([0.0, line_max], [0.0, line_max], color="0.2", linewidth=0.8)
    axis.set(xlim=(0.0, line_max), ylim=(0.0, line_max), xlabel="Satellite wind speed (m s$^{-1}$)", ylabel="TC-RADAR wind speed (m s$^{-1}$)")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    _style_axes(axis)
    _panel_label(axis, "(b) Co-located wind speed")
    axis = axes[1, 0]
    if not radial.empty:
        axis.plot(radial["radius"], radial["fusion_mae"], color="tab:red", linewidth=1.25, label="ML fusion")
        axis.plot(radial["radius"], radial["official_mae"], color="tab:blue", linewidth=1.25, label="Official merge")
    axis.set(xlabel="Radius from satellite center (km)", ylabel="Wind-speed MAE (m s$^{-1}$)")
    axis.legend(frameon=False, fontsize=7, loc="best")
    _style_axes(axis)
    _panel_label(axis, "(c) MAE by radius")
    axis = axes[1, 1]
    axis.axhline(0.0, color="0.2", linewidth=0.8)
    axis.scatter(
        cells["radius_km"],
        cells["absolute_error_reduction_ms"],
        s=8,
        color="tab:green",
        alpha=0.5,
        rasterized=True,
    )
    if not radial.empty:
        axis.plot(radial["radius"], radial["reduction"], color="black", linewidth=1.1)
    axis.set(
        xlabel="Radius from satellite center (km)",
        ylabel="|official error| minus |fusion error| (m s$^{-1}$)",
    )
    _style_axes(axis)
    _panel_label(axis, "(d) Positive values favor ML fusion")
    diagnostic_pdf = output / f"fig_satellite_validation_{mission_id}{figure_tag}_diagnostics.pdf"
    diagnostic_png = output / f"fig_satellite_validation_{mission_id}{figure_tag}_diagnostics.png"
    figure.savefig(diagnostic_pdf, bbox_inches="tight")
    figure.savefig(diagnostic_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "field_pdf": field_pdf,
        "field_png": field_png,
        "diagnostic_pdf": diagnostic_pdf,
        "diagnostic_png": diagnostic_png,
    }


_PROVENANCE_SCOPE_LABELS = {
    "all_common_support": "All common",
    "reference_dominant": "Reference",
    "candidate_dominant": "Candidate",
    "one_source_dominant": "One source",
    "multiple_source_dominant": "Multiple sources",
}


def _radial_profile_frame(cells: pd.DataFrame, bin_width_km: float = 15.0) -> pd.DataFrame:
    """Return observed-common-support radial profiles used in the structure figure."""

    if cells.empty:
        return pd.DataFrame(columns=["radius_km", "satellite", "fusion", "official", "area_weight_km2"])
    maximum = float(np.nanmax(cells["radius_km"].to_numpy(dtype=float)))
    edges = np.arange(0.0, maximum + bin_width_km * 1.01, bin_width_km)
    rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = cells[(cells["radius_km"] >= low) & (cells["radius_km"] < high)]
        if len(selected) < 3:
            continue
        weight = selected["area_weight_km2"].to_numpy(dtype=float)
        rows.append(
            {
                "radius_km": 0.5 * (low + high),
                "satellite": _weighted_mean(selected["satellite_speed_ms"].to_numpy(dtype=float), weight),
                "fusion": _weighted_mean(selected["fused_speed_ms"].to_numpy(dtype=float), weight),
                "official": _weighted_mean(selected["official_speed_ms"].to_numpy(dtype=float), weight),
                "area_weight_km2": float(np.sum(weight)),
            }
        )
    return pd.DataFrame(rows)


def make_satellite_structure_figure(
    cells: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
    mission_id: str,
    storm_name: str,
    satellite_time: pd.Timestamp,
    reference_time: pd.Timestamp,
    resolution_km: float,
) -> dict[str, Path]:
    """Plot the external structural diagnostics before pointwise error measures."""

    import matplotlib.pyplot as plt

    style = Path(__file__).with_name("publication.mplstyle")
    if style.is_file():
        plt.style.use(style)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cells.empty:
        raise ValueError("Cannot draw a satellite structure figure without common-support cells")
    order = [
        "all_common_support",
        "reference_dominant",
        "candidate_dominant",
        "one_source_dominant",
        "multiple_source_dominant",
    ]
    scoped = metrics.set_index("scope").reindex(order)
    labels = [_PROVENANCE_SCOPE_LABELS[name] for name in order]
    profile = _radial_profile_frame(cells)
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.2), constrained_layout=True)
    axis = axes[0, 0]
    if not profile.empty:
        axis.plot(profile["radius_km"], profile["satellite"], color="black", marker="o", ms=2.6, lw=1.2, label="SAR")
        axis.plot(profile["radius_km"], profile["fusion"], color="#D55E00", marker="o", ms=2.6, lw=1.2, label="Reference-time product")
        axis.plot(profile["radius_km"], profile["official"], color="#0072B2", marker="o", ms=2.6, lw=1.2, label="Official merge")
    axis.set(xlabel="Radius from SAR center (km)", ylabel="Wind speed (m s$^{-1}$)", title="Observed radial speed structure")
    axis.legend(frameon=False, fontsize=6.8, loc="best")
    _style_axes(axis)
    _panel_label(axis, "(a)")

    position = np.arange(len(order))
    width = 0.35
    axis = axes[0, 1]
    axis.bar(position - width / 2, scoped["fused_correlation"], width, color="#D55E00", label="Reference-time product")
    axis.bar(position + width / 2, scoped["official_correlation"], width, color="#0072B2", label="Official merge")
    axis.set(xticks=position, xticklabels=labels, ylim=(-0.05, 1.05), ylabel="Spatial correlation with SAR", title="Two-dimensional pattern correlation")
    axis.tick_params(axis="x", labelrotation=22)
    axis.legend(frameon=False, fontsize=6.8, loc="best")
    _style_axes(axis)
    _panel_label(axis, "(b)")

    axis = axes[1, 0]
    axis.bar(position - width / 2, scoped["fused_radial_profile_rmse_ms"], width, color="#D55E00", label="Reference-time product")
    axis.bar(position + width / 2, scoped["official_radial_profile_rmse_ms"], width, color="#0072B2", label="Official merge")
    axis.set(xticks=position, xticklabels=labels, ylabel="Radial-profile RMSE (m s$^{-1}$)", title="Radial profile agreement")
    axis.tick_params(axis="x", labelrotation=22)
    _style_axes(axis)
    _panel_label(axis, "(c)")

    axis = axes[1, 1]
    if not profile.empty:
        axis.scatter(profile["radius_km"], profile["satellite"], color="black", marker="o", s=24, label="SAR")
        axis.scatter(profile["radius_km"], profile["fusion"], color="#D55E00", marker="s", s=24, label="Reference-time product")
        axis.scatter(profile["radius_km"], profile["official"], color="#0072B2", marker="^", s=24, label="Official merge")
        peak = profile.iloc[int(np.nanargmax(profile["satellite"].to_numpy(dtype=float)))]
        axis.axvline(float(peak["radius_km"]), color="0.45", ls="--", lw=0.8)
    axis.set(xlabel="Radius from SAR center (km)", ylabel="Profile wind speed (m s$^{-1}$)", title="Observed profile peak location")
    axis.legend(frameon=False, fontsize=6.8, loc="best")
    _style_axes(axis)
    _panel_label(axis, "(d)")
    offset_minutes = (satellite_time - reference_time).total_seconds() / 60.0
    figure.text(
        0.5,
        -0.015,
        (
            f"{storm_name} {mission_id}; {resolution_km:g}-km area-weighted super-observations; "
            f"SAR {satellite_time:%Y-%m-%d %H:%M:%SZ}; reference {reference_time:%H:%MZ} ({offset_minutes:+.1f} min). "
            "All provenance strata are fixed from the archived fusion product before comparison with SAR."
        ),
        ha="center",
        va="top",
        fontsize=6.35,
        wrap=True,
    )
    pdf = output / f"fig_satellite_validation_{mission_id}_structure.pdf"
    png = output / f"fig_satellite_validation_{mission_id}_structure.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"structure_pdf": pdf, "structure_png": png}


def _format_metric(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def _metrics_at_resolution(metrics: pd.DataFrame, resolution_km: float) -> pd.DataFrame:
    order = [
        "all_common_support",
        "reference_dominant",
        "candidate_dominant",
        "one_source_dominant",
        "multiple_source_dominant",
    ]
    part = metrics[np.isclose(metrics["resolution_km"].astype(float), float(resolution_km))].copy()
    return part.set_index("scope").reindex(order)


def _markdown_structure_table(metrics: pd.DataFrame, resolution_km: float) -> str:
    part = _metrics_at_resolution(metrics, resolution_km)
    lines = [
        "| Provenance stratum | N | Fusion spatial r | Official spatial r | Fusion profile RMSE | Official profile RMSE | Fusion peak-speed abs. error | Official peak-speed abs. error | Fusion peak-radius abs. error | Official peak-radius abs. error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, row in part.iterrows():
        lines.append(
            f"| {_PROVENANCE_SCOPE_LABELS[scope]} | {int(row.get('n_superobservations', 0))} | "
            f"{_format_metric(row.get('fused_correlation'))} | {_format_metric(row.get('official_correlation'))} | "
            f"{_format_metric(row.get('fused_radial_profile_rmse_ms'))} | {_format_metric(row.get('official_radial_profile_rmse_ms'))} | "
            f"{_format_metric(row.get('fused_profile_peak_speed_abs_error_ms'))} | {_format_metric(row.get('official_profile_peak_speed_abs_error_ms'))} | "
            f"{_format_metric(row.get('fused_profile_peak_radius_abs_error_km'))} | {_format_metric(row.get('official_profile_peak_radius_abs_error_km'))} |"
        )
    return "\n".join(lines)


def _markdown_error_table(metrics: pd.DataFrame, resolution_km: float) -> str:
    part = _metrics_at_resolution(metrics, resolution_km)
    lines = [
        "| Provenance stratum | Fusion MAE | Official MAE | Official minus fusion MAE | 15-km block 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, row in part.iterrows():
        interval = (
            f"[{_format_metric(row.get('mae_reduction_block_bootstrap_ci95_low_ms'))}, "
            f"{_format_metric(row.get('mae_reduction_block_bootstrap_ci95_high_ms'))}]"
        )
        lines.append(
            f"| {_PROVENANCE_SCOPE_LABELS[scope]} | {_format_metric(row.get('fused_mae_ms'))} | "
            f"{_format_metric(row.get('official_mae_ms'))} | {_format_metric(row.get('mae_reduction_official_minus_fused_ms'))} | {interval} |"
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    mission_id: str,
    storm_name: str,
    satellite: SatellitePolarWind,
    reference_time: pd.Timestamp,
    reference_start: pd.Timestamp,
    reference_end: pd.Timestamp,
    official_time: pd.Timestamp,
    metrics: pd.DataFrame,
    primary_resolution_km: float,
) -> None:
    satellite_minus_reference = (satellite.measurement_time_utc - reference_time).total_seconds() / 60.0
    official_minus_satellite = (official_time - satellite.measurement_time_utc).total_seconds() / 60.0
    inside_reference_window = reference_start <= satellite.measurement_time_utc <= reference_end
    text = f"""# External satellite validation: {storm_name} {mission_id}

## Design

The STAR SAR product supplies an ocean 10-m wind-speed field at
{satellite.measurement_time_utc:%Y-%m-%d %H:%M:%SZ}. It is compared with
Earth-relative speed reconstructed from the 0.5-km TC-RADAR fusion product and
with the native Earth-relative 0.5-km official merged field. The satellite
product has nominal 3-km resolution; its 1-km radial and 1-degree polar nodes
were area-weighted into {primary_resolution_km:g}-km Cartesian
super-observations. Both TC-RADAR fields were bilinearly sampled at the
original satellite geographic locations before aggregation. Missing field
corners were not bridged, and metrics use only the three-product common
support.

The satellite file does not contain a per-pixel quality flag. Its QC therefore
retains finite values in the physical range
{SATELLITE_MIN_SPEED_MS:g} < wind speed <= {SATELLITE_MAX_SPEED_MS:g} m s-1.
This is an external near-surface comparison, not a claim that a 10-m satellite
retrieval is a three-dimensional truth field.

## Timing

- Satellite scan: {satellite.measurement_time_utc:%Y-%m-%d %H:%M:%SZ}
- Fixed fusion reference midpoint: {reference_time:%Y-%m-%d %H:%M:%SZ}
  ({satellite_minus_reference:+.1f} min from satellite)
- Reference swath observing window: {reference_start:%H:%MZ} to
  {reference_end:%H:%MZ}; satellite inside window: {inside_reference_window}
- Official merged mission time: {official_time:%Y-%m-%d %H:%M:%SZ}
  ({official_minus_satellite:+.1f} min from satellite)

## Structure-first comparison at {primary_resolution_km:g} km

All five provenance strata are defined from the archived fusion output before
any SAR comparison. A cell is ``reference-dominant``, ``one-source-dominant``,
or ``multiple-source-dominant`` when at least half of its co-located satellite
area has provenance 1, 2, or 3, respectively. ``candidate-dominant`` combines
provenance 2 and 3. The structural quantities below are evaluated from the
same area-weighted common-support super-observations: two-dimensional spatial
correlation, 15-km radial-profile RMSE, and the profile peak speed and radial
location. The profile peak is an observed-footprint diagnostic, not an RMW
estimate.

{_markdown_structure_table(metrics, primary_resolution_km)}

## Secondary pointwise-error comparison

{_markdown_error_table(metrics, primary_resolution_km)}

The external comparison is descriptive for one matched near-surface scan. Its
scientific role is to test whether the fixed-reference product retains the
satellite-observed structure at a known time, while retaining the official
merge as the standard mission-analysis baseline. It does not make the 10-m
SAR retrieval a three-dimensional truth field or turn this training-storm
mission into an unseen-storm generalization test.
"""
    path.write_text(text, encoding="utf-8")


def _write_latex_table(
    path: Path,
    metrics: pd.DataFrame,
    resolution_km: float,
) -> None:
    """Write all provenance strata with structural metrics before MAE."""

    part = metrics[np.isclose(metrics["resolution_km"].astype(float), float(resolution_km))].copy()
    order = [
        "all_common_support",
        "reference_dominant",
        "candidate_dominant",
        "one_source_dominant",
        "multiple_source_dominant",
    ]
    part = part.set_index("scope").reindex(order)
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Provenance stratum & $N$ & $r_{\rm F}$ & $r_{\rm O}$ & $P_{\rm F}$ & $P_{\rm O}$ & $|\Delta V_{p,\rm F}|$ & $|\Delta V_{p,\rm O}|$ & $|\Delta r_{p,\rm F}|$ & $|\Delta r_{p,\rm O}|$ \\",
        r"& & \multicolumn{2}{c}{spatial corr.} & \multicolumn{2}{c}{profile RMSE (m s$^{-1}$)} & \multicolumn{2}{c}{peak-speed error (m s$^{-1}$)} & \multicolumn{2}{c}{peak-radius error (km)} \\",
        r"\midrule",
    ]
    for scope in order:
        row = part.loc[scope]
        label = _PROVENANCE_SCOPE_LABELS[scope]
        lines.append(
            f"{label} & {int(row.get('n_superobservations', 0))} & "
            f"{_format_metric(row.get('fused_correlation'))} & {_format_metric(row.get('official_correlation'))} & "
            f"{_format_metric(row.get('fused_radial_profile_rmse_ms'))} & {_format_metric(row.get('official_radial_profile_rmse_ms'))} & "
            f"{_format_metric(row.get('fused_profile_peak_speed_abs_error_ms'))} & {_format_metric(row.get('official_profile_peak_speed_abs_error_ms'))} & "
            f"{_format_metric(row.get('fused_profile_peak_radius_abs_error_km'))} & {_format_metric(row.get('official_profile_peak_radius_abs_error_km'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_satellite_validation(
    satellite_file: str | Path,
    config_path: str | Path,
    mission_id: str = "241107H1",
    resolution_km: float = 3.0,
    min_cell_coverage: float = 0.5,
    bootstrap_resamples: int = 2000,
    fused_product_dir: str | Path | None = None,
    qc_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a fixed-product, resolution-aware external satellite comparison."""

                                                                             
                                                    
    config_candidate = Path(config_path).expanduser().resolve()
    try:
        config_payload = json.loads(config_candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config_payload = {}
    if isinstance(config_payload, dict) and "base_config" in config_payload:
        cfg = _load_overlay_config_without_torch(config_candidate)
    else:
        cfg = load_config(config_candidate)
    root = Path(cfg["paths"]["output_root"])
    output = Path(cfg["paths"].get("satellite_validation_dir", root / "satellite_validation")) / str(mission_id)
    figure_dir = Path(
        cfg["paths"].get("satellite_validation_figure_dir", root / "figures" / "satellite_validation")
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    qc_base = Path(qc_root) if qc_root is not None else root
    merged = pd.read_csv(qc_base / "qc" / "fusion_qc_merged.csv")
    swath = pd.read_csv(qc_base / "qc" / "fusion_qc_swath.csv")
    selected = merged[(merged["mission_ID"].astype(str) == str(mission_id)) & merged["fusion_qc_pass"].astype(bool)]
    if len(selected) != 1:
        raise ValueError(f"Expected one quality-control passing merged record for {mission_id}; found {len(selected)}")
    merged_row = selected.iloc[0]
    reference_rows = swath[
        (swath["mission_ID"].astype(str) == str(mission_id))
        & swath["swath_qc_pass"].astype(bool)
        & swath["parent_mission_fusion_qc_pass"].astype(bool)
        & swath["is_primary_reference"].astype(bool)
    ]
    if len(reference_rows) != 1:
        raise ValueError(f"Expected one fixed primary reference for {mission_id}; found {len(reference_rows)}")
    reference_row = reference_rows.iloc[0]
    satellite = read_star_sar_polar(satellite_file)
    height_km = float(cfg["domain"].get("height_min_km", 0.5))
    reference_frame = read_reference_frame(reference_row, height_km)
    fused_path = Path(fused_product_dir) if fused_product_dir is not None else Path(cfg["paths"]["archive_product_dir"])
    fused_path = fused_path / f"MLNEW_FUSED_{mission_id}.nc"
    if not fused_path.is_file():
        raise FileNotFoundError(f"Fixed archive product is missing: {fused_path}")
    fused = read_fused_product(fused_path, reference_frame, height_km)
    official = read_official_product(merged_row, height_km)
    reference_time = _utc_timestamp(reference_row["swath_midpoint_time"])
    reference_start = _utc_timestamp(reference_row["swath_start_time"])
    reference_end = _utc_timestamp(reference_row["swath_end_time"])
    official_time = _utc_timestamp(merged_row["mission_time"])
    radius_limit_km = float(cfg["domain"]["radius_max_km"])
    resolutions = sorted({2.0, float(resolution_km), 4.0})
    metrics_records: list[dict[str, Any]] = []
    primary_cells = pd.DataFrame()
    primary_nodes = pd.DataFrame()
    for current_resolution in resolutions:
        nodes = build_satellite_superobservations(
            satellite,
            current_resolution,
            radius_limit_km,
            min_cell_coverage,
        )
        colocated = colocate_satellite_nodes(nodes, fused, official)
        cells = aggregate_common_superobservations(colocated, min_cell_coverage)
        for scope, subset in _scope_tables(cells):
            record = summarize_external_metrics(
                subset,
                scope,
                seed=int(cfg["project"]["seed"]) + int(round(1000.0 * current_resolution)),
                bootstrap_resamples=bootstrap_resamples,
            )
            record["resolution_km"] = current_resolution
            metrics_records.append(record)
        if math.isclose(current_resolution, float(resolution_km), abs_tol=1.0e-8):
            primary_nodes = colocated
            primary_cells = cells
    metrics = pd.DataFrame(metrics_records).sort_values(["resolution_km", "scope"], kind="mergesort")
    metrics_path = output / "satellite_validation_metrics_by_resolution_and_scope.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    primary_cells_path = output / f"{mission_id}_satellite_common_superobservations_{resolution_km:g}km.csv"
    primary_cells.to_csv(primary_cells_path, index=False, encoding="utf-8")
    primary_nodes_path = output / f"{mission_id}_satellite_colocated_native_nodes_{resolution_km:g}km.csv"
    primary_nodes.to_csv(primary_nodes_path, index=False, encoding="utf-8")
    npz_path = output / f"{mission_id}_satellite_validation_{resolution_km:g}km.npz"
    np.savez_compressed(
        npz_path,
        **{
            column: primary_cells[column].to_numpy()
            for column in primary_cells.columns
        },
    )
    primary_metrics = metrics[
        (np.isclose(metrics["resolution_km"].astype(float), float(resolution_km)))
        & (metrics["scope"] == "all_common_support")
    ]
    if len(primary_metrics) != 1:
        raise RuntimeError("Primary all-common-support summary was not generated")
    primary = primary_metrics.iloc[0]
    figures = make_satellite_validation_figures(
        primary_cells,
        metrics[np.isclose(metrics["resolution_km"].astype(float), float(resolution_km))],
        figure_dir,
        str(mission_id),
        str(merged_row["storm_name"]),
        satellite.measurement_time_utc,
        reference_time,
        official_time,
        float(resolution_km),
        radius_limit_km,
    )
    figures.update(
        make_satellite_structure_figure(
            primary_cells,
            metrics[np.isclose(metrics["resolution_km"].astype(float), float(resolution_km))],
            figure_dir,
            str(mission_id),
            str(merged_row["storm_name"]),
            satellite.measurement_time_utc,
            reference_time,
            float(resolution_km),
        )
    )
    multiple_metrics = metrics[
        (np.isclose(metrics["resolution_km"].astype(float), float(resolution_km)))
        & (metrics["scope"] == "multiple_source_dominant")
    ]
    if len(multiple_metrics) != 1:
        raise RuntimeError("Primary multiple-source summary was not generated")
    multiple = multiple_metrics.iloc[0]
    multiple_cells = primary_cells[primary_cells["multiple_source_fraction"] >= 0.5].copy()
    if not multiple_cells.empty:
        multiple_figures = make_satellite_validation_figures(
            multiple_cells,
            metrics[np.isclose(metrics["resolution_km"].astype(float), float(resolution_km))],
            figure_dir,
            str(mission_id),
            str(merged_row["storm_name"]),
            satellite.measurement_time_utc,
            reference_time,
            official_time,
            float(resolution_km),
            radius_limit_km,
            metric_scope="multiple_source_dominant",
            figure_tag="_multiple_source",
            scope_label="multiple-source candidate-dominant cells",
        )
        figures.update({f"multiple_source_{key}": value for key, value in multiple_figures.items()})
    report_path = output / "SATELLITE_VALIDATION_REPORT.md"
    _write_report(
        report_path,
        str(mission_id),
        str(merged_row["storm_name"]),
        satellite,
        reference_time,
        reference_start,
        reference_end,
        official_time,
        metrics,
        float(resolution_km),
    )
    latex_table_path = output / "table_satellite_comparison.tex"
    _write_latex_table(latex_table_path, metrics, float(resolution_km))
    metadata = {
        "mission_ID": str(mission_id),
        "storm_key": str(merged_row["storm_key"]),
        "storm_name": str(merged_row["storm_name"]),
        "split": str(merged_row["split"]),
        "comparison_type": (
            "external 10-m satellite wind speed versus Earth-relative 0.5-km "
            "TC-RADAR products on common satellite footprint; descriptive single-case diagnostic"
        ),
        "satellite_file": str(satellite.path),
        "satellite_filename": satellite.path.name,
        "satellite_sha256": _sha256(satellite.path),
        "satellite_measurement_time_utc": satellite.measurement_time_utc.isoformat(),
        "satellite_center_latitude": satellite.center_latitude,
        "satellite_center_longitude": satellite.center_longitude,
        "satellite_source_attributes": satellite.source_attributes,
        "satellite_qc": {
            "per_pixel_quality_flag_present": False,
            "valid_rule": "finite latitude, longitude, and wind_speed; 0 < wind_speed <= 100 m s-1",
            "native_valid_wind_nodes": int(np.count_nonzero(satellite.valid)),
            "native_total_nodes": int(satellite.valid.size),
        },
        "target_height_km": height_km,
        "target_resolution_km": float(resolution_km),
        "resolution_sensitivity_km": resolutions,
        "min_satellite_cell_coverage": float(min_cell_coverage),
        "min_model_common_coverage": float(min_cell_coverage),
        "radius_limit_km": radius_limit_km,
        "geographic_colocation": (
            "inverse piecewise-linear mapping of each product's stored recentered latitude/longitude grid; "
            "finite-only bilinear field interpolation at satellite native locations"
        ),
        "aggregation": (
            "annular-sector area weighting of native satellite polar nodes into Cartesian super-observations"
        ),
        "reference_swath_ID": str(reference_row["swath_ID"]),
        "reference_midpoint_time_utc": reference_time.isoformat(),
        "reference_start_time_utc": reference_start.isoformat(),
        "reference_end_time_utc": reference_end.isoformat(),
        "official_merge_time_utc": official_time.isoformat(),
        "satellite_minus_reference_midpoint_minutes": float(
            (satellite.measurement_time_utc - reference_time).total_seconds() / 60.0
        ),
        "official_minus_satellite_minutes": float(
            (official_time - satellite.measurement_time_utc).total_seconds() / 60.0
        ),
        "satellite_inside_reference_swath_window": bool(reference_start <= satellite.measurement_time_utc <= reference_end),
        "fusion_storm_to_earth_motion_eastward_ms": reference_frame.motion_eastward_ms,
        "fusion_storm_to_earth_motion_northward_ms": reference_frame.motion_northward_ms,
        "fusion_storm_to_earth_motion_max_abs_deviation_ms": reference_frame.motion_component_max_abs_deviation_ms,
        "archive_product": str(fused_path),
        "official_merged_source": str(merged_row["source_path"]),
        "outputs": {
            "metrics": str(metrics_path),
            "primary_superobservations": str(primary_cells_path),
            "primary_colocated_native_nodes": str(primary_nodes_path),
            "primary_npz": str(npz_path),
            "report": str(report_path),
            "latex_table": str(latex_table_path),
            **{key: str(value) for key, value in figures.items()},
        },
    }
    metadata_path = output / "satellite_validation_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "mission_ID": str(mission_id),
        "storm_name": str(merged_row["storm_name"]),
        "satellite_time_utc": satellite.measurement_time_utc.isoformat(),
        "reference_time_utc": reference_time.isoformat(),
        "official_time_utc": official_time.isoformat(),
        "primary_resolution_km": float(resolution_km),
        "primary_metrics": {key: value for key, value in primary.to_dict().items()},
        "outputs": metadata["outputs"],
        "manifest": str(metadata_path),
    }
