from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import polar_geometry


def stable_mask_seed(base_seed: int, mission_id: str, mask_index: int, epoch: int = 0) -> int:
    token = f"{base_seed}|{mission_id}|{mask_index}|{epoch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def geometric_mask(
    shape: tuple[int, int, int],
    x_km: np.ndarray,
    y_km: np.ndarray,
    rng: np.random.Generator,
    mask_type: str,
    fraction_range: tuple[float, float],
) -> np.ndarray:
    nz, ny, nx = shape
    radius, theta = polar_geometry(x_km, y_km)
    target = float(rng.uniform(*fraction_range))
    if mask_type == "sector":
        width = np.clip(2 * np.pi * target * rng.uniform(0.8, 1.5), np.deg2rad(15), np.deg2rad(150))
        center = rng.uniform(-np.pi, np.pi)
        diff = np.angle(np.exp(1j * (theta - center)))
        mask2 = np.abs(diff) <= width / 2
        z0 = int(rng.integers(0, max(1, nz // 3)))
        z1 = int(rng.integers(max(z0 + 1, 2 * nz // 3), nz + 1))
        return np.broadcast_to(mask2, shape) & (np.arange(nz)[:, None, None] >= z0) & (np.arange(nz)[:, None, None] < z1)
    if mask_type == "stripe":
        angle = rng.uniform(0, np.pi)
        xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
        cross = xx * np.cos(angle) + yy * np.sin(angle)
        width = max(np.ptp(cross) * target, np.median(np.diff(x_km)) * 3)
        center = rng.uniform(np.nanmin(cross), np.nanmax(cross))
        return np.broadcast_to(np.abs(cross - center) <= width / 2, shape).copy()
    if mask_type == "block":
        sy = max(3, int(round(ny * np.sqrt(target))))
        sx = max(3, int(round(nx * np.sqrt(target))))
        y0 = int(rng.integers(0, max(1, ny - sy + 1)))
        x0 = int(rng.integers(0, max(1, nx - sx + 1)))
        zsize = max(1, int(round(nz * rng.uniform(0.4, 1.0))))
        z0 = int(rng.integers(0, max(1, nz - zsize + 1)))
        out = np.zeros(shape, dtype=bool)
        out[z0 : z0 + zsize, y0 : y0 + sy, x0 : x0 + sx] = True
        return out
    if mask_type == "radial_gap":
        rmax = float(np.nanmax(radius))
        width = max(rmax * target, np.median(np.diff(x_km)) * 4)
        center = rng.uniform(width / 2, max(width / 2, rmax - width / 2))
        mask2 = np.abs(radius - center) <= width / 2
        return np.broadcast_to(mask2, shape).copy()
    if mask_type == "mixed":
        first = geometric_mask(shape, x_km, y_km, rng, "sector", (target / 2, target))
        second = geometric_mask(shape, x_km, y_km, rng, "block", (target / 3, target * 0.8))
        return first | second
    raise ValueError(f"Unknown mask type: {mask_type}")


def coverage_aware_hidden_mask(
    geometry: np.ndarray,
    reference_observed: np.ndarray,
    warped_support: np.ndarray,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    support_count = np.sum(np.asarray(warped_support) >= float(support_threshold), axis=0)
    hidden = np.asarray(geometry, dtype=bool) & np.asarray(reference_observed, dtype=bool) & (support_count >= 1)
    return hidden, support_count.astype(np.int16)


def apply_reference_mask(u: np.ndarray, v: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(u) & np.isfinite(v)
    hidden = np.asarray(hidden, dtype=bool) & observed
    visible_u = np.asarray(u, dtype=np.float32).copy()
    visible_v = np.asarray(v, dtype=np.float32).copy()
    visible_u[hidden] = np.nan
    visible_v[hidden] = np.nan
    visible = observed & ~hidden
    return visible_u, visible_v, visible


def build_fixed_mask_manifest(merged: pd.DataFrame, cfg: dict[str, Any], output_root: str | Path) -> Path:
    out = Path(output_root) / "masks"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    mask_cfg = cfg["mask"]
    for _, row in merged[merged["fusion_qc_pass"] & merged["split"].isin(["validation", "test"])].sort_values("mission_ID").iterrows():
        for index in range(int(mask_cfg["fixed_masks_per_mission"])):
            seed = stable_mask_seed(int(mask_cfg["seed"]), str(row["mission_ID"]), index)
            mask_types = list(mask_cfg["types"])
            rows.append({
                "mission_ID": row["mission_ID"],
                "storm_key": row["storm_key"],
                "split": row["split"],
                "mask_index": index,
                "seed": seed,
                "mask_type": mask_types[index % len(mask_types)],
                "support_threshold": float(mask_cfg["support_threshold"]),
            })
    path = out / "fixed_mask_bank.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    metadata = {
        "definition": "G intersect reference_observed intersect candidate_support_count>=1",
        "seed": int(mask_cfg["seed"]),
        "dynamic_training_masks": True,
        "fixed_validation_test_masks": True,
    }
    (out / "fixed_mask_bank_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path

