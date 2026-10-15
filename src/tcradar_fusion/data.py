"""Masked-reference pairs for fixed-coordinate temporal harmonization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_io import _attach_cache_paths, _tables
from .data_io import MissionArrays, _load_mission
from .geometry import polar_geometry, uv_to_radial_tangential
from .masking import geometric_mask

from .multiscale import decompose_source


@dataclass(frozen=True)
class PairRecord:
    mission_id: str
    storm_key: str
    split: str
    pseudo_reference_index: int
    mask_index: int
    mask_seed: int
    mask_type: str


def pair_seed(seed: int, mission_id: str, ref_index: int, mask_index: int, epoch: int = 0) -> int:
    token = f"tsc-ntf|{seed}|{mission_id}|{ref_index}|{mask_index}|{epoch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def _visible_components(u: np.ndarray, v: np.ndarray, mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ur, ut = uv_to_radial_tangential(np.where(mask, u, np.nan), np.where(mask, v, np.nan), x, y)
    ur = np.nan_to_num(ur, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ut = np.nan_to_num(ut, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ur[~mask] = 0.0
    ut[~mask] = 0.0
    return ur, ut


def _distance_to_edge(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    result = np.zeros(mask.shape, dtype=np.float32)
    for iz in range(mask.shape[0]):
        result[iz] = distance_transform_edt(mask[iz]).astype(np.float32)
    scale = max(float(np.nanpercentile(result, 90)) if np.any(result > 0) else 1.0, 1.0)
    return np.clip(result / scale, 0.0, 1.0)


class MissionStore:
    """Read audited QC products and lazily load immutable mission arrays."""

    def __init__(self, parent_cfg: dict[str, Any]) -> None:
        merged, swath = _tables(parent_cfg)
        swath = _attach_cache_paths(parent_cfg, swath)
        selected = merged[merged["fusion_qc_pass"].astype(bool)].copy()
        self.parent_cfg = parent_cfg
        self.merged = selected.set_index("mission_ID").sort_index()
        self.groups: dict[str, pd.DataFrame] = {}
        for mission_id, group in swath[swath["mission_ID"].isin(self.merged.index)].groupby("mission_ID", sort=True):
            good = group[group["swath_qc_pass"].astype(bool) & group["parent_mission_fusion_qc_pass"].astype(bool)].copy()
            self.groups[str(mission_id)] = good.sort_values(["source_file", "swath_case_index"], kind="mergesort").reset_index(drop=True)
        self._missions: dict[str, MissionArrays] = {}
        self._decompositions: dict[tuple[str, int], dict[str, np.ndarray | float]] = {}
        fixed_path = Path(parent_cfg["paths"]["output_root"]) / "masks" / "fixed_mask_bank.csv"
        self.fixed_masks = pd.read_csv(fixed_path) if fixed_path.exists() else pd.DataFrame()

    def mission_ids(self, storms: set[str] | None = None) -> list[str]:
        result: list[str] = []
        for mission_id, row in self.merged.iterrows():
            if storms is not None and str(row["storm_key"]) not in storms:
                continue
            if mission_id in self.groups and len(self.groups[mission_id]) >= 2:
                result.append(str(mission_id))
        return sorted(result)

    def mission(self, mission_id: str) -> MissionArrays:
        mission_id = str(mission_id)
        if mission_id not in self._missions:
            row = self.merged.loc[mission_id].copy()
            row.name = mission_id
            self._missions[mission_id] = _load_mission(self.groups[mission_id], row, self.parent_cfg)
        return self._missions[mission_id]

    def component(self, mission_id: str, swath_index: int, cfg: dict[str, Any]) -> dict[str, np.ndarray | float]:
        key = (str(mission_id), int(swath_index))
        if key not in self._decompositions:
            mission = self.mission(mission_id)
            support = np.isfinite(mission.u[swath_index]) & np.isfinite(mission.v[swath_index])
            comp_cfg = {
                "radius_max_km": float(cfg["data"]["radius_max_km"]),
                "radial_bin_km": float(cfg["data"]["radial_bin_km"]),
                "large_scale_sigma_km": float(cfg["data"]["large_scale_sigma_km"]),
                "gaussian_mask_threshold": float(cfg["data"]["gaussian_mask_threshold"]),
                "min_annular_points": int(cfg["data"]["min_annular_points"]),
            }
            self._decompositions[key] = decompose_source(mission.u[swath_index], mission.v[swath_index], support, mission.x, mission.y, comp_cfg)
        return self._decompositions[key]

    def records(self, storms: set[str] | None, split: str, cfg: dict[str, Any], epoch: int = 0) -> list[PairRecord]:
        masks = cfg["mask"]
        count_key = "masks_per_pair_train" if split == "train" else "masks_per_pair_validation"
        if split in {"locked_test", "test"}:
            count_key = "masks_per_pair_test"
        n_masks = int(masks.get(count_key, 1))
        rows: list[PairRecord] = []
        for mission_id in self.mission_ids(storms):
            mission = self.mission(mission_id)
            storm_key = str(self.merged.loc[mission_id, "storm_key"])
            for ref_index in range(len(mission.swath_ids)):
                                                                           
                                                                         
                                                                            
                                                   
                fixed_rows = (
                    self.fixed_masks[self.fixed_masks["mission_ID"].astype(str) == str(mission_id)]
                    if split not in {"train", "archive", "archive_inference"} and not self.fixed_masks.empty
                    else pd.DataFrame()
                )
                mask_iter = [(int(row["mask_index"]), int(row["seed"]), str(row["mask_type"])) for _, row in fixed_rows.sort_values("mask_index").iterrows()] if not fixed_rows.empty else [(i, pair_seed(int(cfg["project"]["seed"]), mission_id, ref_index, i), str(masks["types"][i % len(masks["types"])])) for i in range(n_masks)]
                for mask_index, fixed_seed, fixed_type in mask_iter:
                    if split in {"archive", "archive_inference"}:
                        rows.append(PairRecord(mission_id, storm_key, split, ref_index, 0, 0, "none"))
                    else:
                        seed = int(fixed_seed) if not fixed_rows.empty else pair_seed(int(cfg["project"]["seed"]), mission_id, ref_index, mask_index, epoch if split == "train" else 0)
                        kind = fixed_type
                        rows.append(PairRecord(mission_id, storm_key, split, ref_index, mask_index, seed, kind))
        return rows


def _overlap_fraction(source: np.ndarray, context: np.ndarray) -> float:
    a = int(source.sum())
    b = int(context.sum())
    return float((source & context).sum() / max(1, min(a, b)))


def build_pair(record: PairRecord, store: MissionStore, cfg: dict[str, Any], normalization: dict[str, Any], epoch: int = 0) -> dict[str, Any]:
    mission = store.mission(record.mission_id)
    ref_index = int(record.pseudo_reference_index)
    ref_u = np.asarray(mission.u[ref_index], dtype=np.float32)
    ref_v = np.asarray(mission.v[ref_index], dtype=np.float32)
    ref_observed = np.isfinite(ref_u) & np.isfinite(ref_v)
    source_indices = [idx for idx in range(len(mission.swath_ids)) if idx != ref_index]
    source_u = np.asarray(mission.u[source_indices], dtype=np.float32)
    source_v = np.asarray(mission.v[source_indices], dtype=np.float32)
    source_support = np.isfinite(source_u) & np.isfinite(source_v)
    union_support = np.any(source_support, axis=0)
    if record.mask_type == "none":
        geometry = np.zeros_like(ref_observed, dtype=bool)
    else:
        rng = np.random.default_rng(int(pair_seed(int(cfg["project"]["seed"]), record.mission_id, ref_index, record.mask_index, epoch if record.split == "train" else 0)))
        geometry = geometric_mask(ref_observed.shape, mission.x, mission.y, rng, record.mask_type, (float(cfg["mask"]["hidden_fraction_min"]), float(cfg["mask"]["hidden_fraction_max"])))
                                                                               
                                                          
    hidden = geometry & ref_observed & union_support
    context = ref_observed & ~hidden
    ref_context_r, ref_context_t = _visible_components(ref_u, ref_v, context, mission.x, mission.y)
                                                                                
    from .multiscale import decompose_source
    ref_ctx_comp = decompose_source(
        np.where(context, ref_u, np.nan), np.where(context, ref_v, np.nan), context, mission.x, mission.y,
        {"radius_max_km": cfg["data"]["radius_max_km"], "radial_bin_km": cfg["data"]["radial_bin_km"], "large_scale_sigma_km": cfg["data"]["large_scale_sigma_km"], "gaussian_mask_threshold": cfg["data"]["gaussian_mask_threshold"], "min_annular_points": cfg["data"]["min_annular_points"]},
    )
    radius, theta = polar_geometry(mission.x, mission.y)
    rnorm = np.broadcast_to(radius[None] / max(float(cfg["data"]["radius_max_km"]), 1.0), ref_u.shape).astype(np.float32)
    z_norm = np.broadcast_to((mission.z / max(float(cfg["data"]["height_max_km"]), 1.0))[:, None, None], ref_u.shape).astype(np.float32)
    sin_theta = np.broadcast_to(np.sin(theta)[None], ref_u.shape).astype(np.float32)
    cos_theta = np.broadcast_to(np.cos(theta)[None], ref_u.shape).astype(np.float32)
    edge_maps = [_distance_to_edge(source_support[i]) for i in range(len(source_indices))]
    u_std = max(float(normalization.get("u_std", 14.0)), 1.0e-6)
    v_std = max(float(normalization.get("v_std", 14.0)), 1.0e-6)
    wind_std = max(0.5 * (u_std + v_std), 1.0e-6)
    features: list[np.ndarray] = []
    components: list[dict[str, np.ndarray | float]] = []
    tau = np.asarray(mission.times_hours[ref_index] - mission.times_hours[source_indices], dtype=np.float32)
    n_earlier = float(np.count_nonzero(tau > 0.0))
    n_later = float(np.count_nonzero(tau < 0.0))
    for i, source_index in enumerate(source_indices):
        comp = store.component(record.mission_id, source_index, cfg)
        components.append(comp)
        sr = np.asarray(comp["source_radial"], dtype=np.float32)
        st = np.asarray(comp["source_tangential"], dtype=np.float32)
        bg_r = np.asarray(comp["background_r"], dtype=np.float32)
        bg_t = np.asarray(comp["background_t"], dtype=np.float32)
        large_r = np.asarray(comp["large_r"], dtype=np.float32)
        large_t = np.asarray(comp["large_t"], dtype=np.float32)
        small_r = np.asarray(comp["small_r"], dtype=np.float32)
        small_t = np.asarray(comp["small_t"], dtype=np.float32)
        overlap = _overlap_fraction(source_support[i], context)
        channels = [
            sr / wind_std, st / wind_std, bg_r / wind_std, bg_t / wind_std,
            large_r / wind_std, large_t / wind_std, small_r / wind_std, small_t / wind_std,
            source_support[i].astype(np.float32), ref_context_r / wind_std, ref_context_t / wind_std,
            context.astype(np.float32),
            np.full_like(ref_u, float(tau[i] / 4.0)), np.full_like(ref_u, float(abs(tau[i]) / 4.0)),
            np.full_like(ref_u, float(np.sign(tau[i]))), rnorm, sin_theta, cos_theta, z_norm,
            np.full_like(ref_u, float(mission.coverages[source_index])), np.full_like(ref_u, float(context.mean())),
            np.full_like(ref_u, overlap),
            (bg_r - np.asarray(ref_ctx_comp["background_r"], dtype=np.float32)) / wind_std,
            (bg_t - np.asarray(ref_ctx_comp["background_t"], dtype=np.float32)) / wind_std,
            np.nan_to_num(np.asarray(comp["annular_valid_fraction_grid"]), nan=0.0),
            np.nan_to_num(np.asarray(comp["annular_uncertainty_grid"]), nan=0.0) / wind_std,
            edge_maps[i],
        ]
        features.append(np.nan_to_num(np.stack(channels), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    target_r, target_t = uv_to_radial_tangential(ref_u, ref_v, mission.x, mission.y)
    target_r = np.nan_to_num(target_r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target_t = np.nan_to_num(target_t, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    result: dict[str, Any] = {
        "features": np.stack(features).astype(np.float32),
        "source_u": np.nan_to_num(source_u, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "source_v": np.nan_to_num(source_v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "source_radial": np.stack([np.asarray(c["source_radial"]) for c in components]).astype(np.float32),
        "source_tangential": np.stack([np.asarray(c["source_tangential"]) for c in components]).astype(np.float32),
        "source_background_r": np.stack([np.asarray(c["background_r"]) for c in components]).astype(np.float32),
        "source_background_t": np.stack([np.asarray(c["background_t"]) for c in components]).astype(np.float32),
        "source_large_r": np.stack([np.asarray(c["large_r"]) for c in components]).astype(np.float32),
        "source_large_t": np.stack([np.asarray(c["large_t"]) for c in components]).astype(np.float32),
        "source_small_r": np.stack([np.asarray(c["small_r"]) for c in components]).astype(np.float32),
        "source_small_t": np.stack([np.asarray(c["small_t"]) for c in components]).astype(np.float32),
        "source_support": source_support.astype(np.float32),
        "reference_context_r": ref_context_r.astype(np.float32), "reference_context_t": ref_context_t.astype(np.float32),
        "reference_mask": context.astype(bool), "reference_target_r": target_r, "reference_target_t": target_t,
        "hidden_mask": hidden.astype(bool), "tau_hours": tau, "source_coverages": np.asarray(mission.coverages[source_indices], dtype=np.float32),
        "candidate_support_count": source_support.sum(axis=0).astype(np.int16),
        "x_km": mission.x.astype(np.float32), "y_km": mission.y.astype(np.float32), "z_km": mission.z.astype(np.float32),
        "radius_grid": radius.astype(np.float32), "sin_theta": sin_theta, "cos_theta": cos_theta, "z_norm": z_norm,
        "mission_id": record.mission_id, "storm_key": record.storm_key, "split": record.split,
        "pseudo_reference_swath_id": str(mission.swath_ids[ref_index]), "source_swath_ids": [str(mission.swath_ids[i]) for i in source_indices],
        "pseudo_reference_index": ref_index, "mask_seed": int(record.mask_seed), "mask_type": record.mask_type,
        "n_earlier": n_earlier, "n_later": n_later,
    }
    if np.any(hidden & ~ref_observed):
        raise AssertionError("hidden target is outside reference observation support")
    if np.any(hidden & ~union_support):
        raise AssertionError("hidden target has no unchanged source support")
    return result


class PairDataset(Dataset):
    def __init__(self, store: MissionStore, records: list[PairRecord], cfg: dict[str, Any], normalization: dict[str, Any]) -> None:
        self.store, self.records, self.cfg, self.normalization = store, records, cfg, normalization
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return build_pair(self.records[index], self.store, self.cfg, self.normalization, self.epoch)


def pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 because source counts vary by mission")
    item = batch[0]
    tensor_keys = (
        "features", "source_u", "source_v", "source_radial", "source_tangential", "source_background_r", "source_background_t",
        "source_large_r", "source_large_t", "source_small_r", "source_small_t", "source_support", "reference_context_r", "reference_context_t",
        "reference_mask", "reference_target_r", "reference_target_t", "hidden_mask", "tau_hours", "source_coverages", "candidate_support_count",
        "x_km", "y_km", "z_km", "radius_grid", "sin_theta", "cos_theta", "z_norm",
    )
    out: dict[str, Any] = {}
    for key in tensor_keys:
        value = torch.from_numpy(np.asarray(item[key]))
        out[key] = value.unsqueeze(0)
    for key in ("mission_id", "storm_key", "split", "pseudo_reference_swath_id", "source_swath_ids", "pseudo_reference_index", "mask_seed", "mask_type", "n_earlier", "n_later"):
        out[key] = [item[key]]
    return out
