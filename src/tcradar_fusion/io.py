from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np

from .geometry import derivatives, radial_tangential_to_uv, uv_to_radial_tangential


def _var(ds: nc.Dataset, name: str, dtype: str, dims: tuple[str, ...], units: str = "") -> nc.Variable:
    var = ds.createVariable(name, dtype, dims, zlib=True, complevel=4, fill_value=np.nan if dtype.startswith("f") else None)
    if units:
        var.units = units
    return var


def write_product(path: str | Path, payload: dict[str, np.ndarray], coordinates: dict[str, np.ndarray], metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u = np.asarray(payload["fused_u"], dtype=np.float32)
    v = np.asarray(payload["fused_v"], dtype=np.float32)
    radial, tangential = uv_to_radial_tangential(u, v, coordinates["x"], coordinates["y"])
    speed = np.hypot(u, v).astype(np.float32)
    vort, div = derivatives(u, v, coordinates["x"], coordinates["y"])
    support = np.asarray(payload["candidate_support"], dtype=np.float32)
    nsource = int(support.shape[0])
    dims = ("z", "y", "x")
    sdims = ("source_swath", "z", "y", "x")
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("source_swath", nsource)
        ds.createDimension("z", len(coordinates["z"]))
        ds.createDimension("y", len(coordinates["y"]))
        ds.createDimension("x", len(coordinates["x"]))
        for name in ("z", "y", "x"):
            coord = ds.createVariable(name, "f4", (name,))
            coord[:] = np.asarray(coordinates[name], dtype=np.float32)
            coord.units = "km"
        fields = {
            "fused_eastward_wind": (u, "m s-1"), "fused_northward_wind": (v, "m s-1"),
            "fused_radial_wind": (radial, "m s-1"), "fused_tangential_wind": (tangential, "m s-1"),
            "fused_wind_speed": (speed, "m s-1"), "fused_relative_vorticity": (vort, "s-1"), "fused_divergence": (div, "s-1"),
            "fused_uncertainty_u": (np.asarray(payload["uncertainty_u"]), "m s-1"),
            "fused_uncertainty_v": (np.asarray(payload["uncertainty_v"]), "m s-1"),
            "fused_uncertainty_speed": (np.asarray(payload["uncertainty_speed"]), "m s-1"),
        }
        for name, (values, units) in fields.items():
            _var(ds, name, "f4", dims, units)[:] = np.asarray(values, dtype=np.float32)
        _var(ds, "reference_u", "f4", dims, "m s-1")[:] = np.asarray(payload["reference_u"], dtype=np.float32)
        _var(ds, "reference_v", "f4", dims, "m s-1")[:] = np.asarray(payload["reference_v"], dtype=np.float32)
        ds.createVariable("reference_mask", "i1", dims)[:] = np.asarray(payload["reference_mask"], dtype=np.int8)
        ds.createVariable("candidate_support_count", "i2", dims)[:] = np.asarray(payload["candidate_support_count"], dtype=np.int16)
        ds.createVariable("effective_source_count", "f4", dims)[:] = np.asarray(payload["effective_source_count"], dtype=np.float32)
        ds.createVariable("fusion_provenance", "i1", dims)[:] = np.asarray(payload["provenance"], dtype=np.int8)
        ds.createVariable("support_provenance", "i1", dims)[:] = np.asarray(payload["provenance"], dtype=np.int8)
        ds.variables["fusion_provenance"].flag_values = np.asarray([0, 1, 2, 3], dtype=np.int8)
        ds.variables["fusion_provenance"].flag_meanings = "no_estimate reference_observation one_corrected_source multiple_corrected_sources"
        for name, values, units in (
            ("candidate_support", support, "1"), ("candidate_weights", payload["weights"], "1"),
            ("candidate_raw_u", payload["candidate_raw_u"], "m s-1"), ("candidate_raw_v", payload["candidate_raw_v"], "m s-1"),
            ("candidate_corrected_u", payload["candidate_corrected_u"], "m s-1"), ("candidate_corrected_v", payload["candidate_corrected_v"], "m s-1"),
            ("candidate_increment_B_r", payload["increment_b_r"], "m s-1"), ("candidate_increment_B_t", payload["increment_b_t"], "m s-1"),
            ("candidate_increment_L_r", payload["increment_l_r"], "m s-1"), ("candidate_increment_L_t", payload["increment_l_t"], "m s-1"),
            ("candidate_increment_S_r", payload["increment_s_r"], "m s-1"), ("candidate_increment_S_t", payload["increment_s_t"], "m s-1"),
            ("candidate_sigma_r", payload["sigma_r"], "m s-1"), ("candidate_sigma_t", payload["sigma_t"], "m s-1"),
            ("candidate_rho_rt", payload["rho"], "1"), ("candidate_quality", payload["quality"], "1"),
        ):
            _var(ds, name, "f4", sdims, units)[:] = np.asarray(values, dtype=np.float32)
        _var(ds, "fused_covariance_rt", "f4", dims, "m2 s-2")[:] = np.asarray(payload["covariance_rt"], dtype=np.float32)
        for name, values, units in (("source_signed_tau", payload["tau_hours"], "hour"), ("source_time_hours", payload["source_time_hours"], "hour")):
            variable = ds.createVariable(name, "f4", ("source_swath",))
            variable[:] = np.asarray(values, dtype=np.float32)
            variable.units = units
        source_ids = ds.createVariable("source_swath_id", str, ("source_swath",))
        source_ids[:] = np.asarray(payload["source_swath_ids"], dtype=object)
        ref_id = ds.createVariable("reference_swath_id", str, ())
        ref_id[()] = str(payload["reference_swath_id"])
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True, default=str)
            ds.setncattr(str(key), str(value))
        ds.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
        ds.setncattr("coordinate_contract", "x_source=x_target; y_source=y_target; r_source=r_target; theta_source=theta_target; z_source=z_target")
        ds.setncattr("support_contract", "candidate_support equals source_observation_support at identical grid cells")
    return path
