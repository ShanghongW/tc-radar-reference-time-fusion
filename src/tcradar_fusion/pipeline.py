from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import MissionStore, PairDataset, PairRecord, pair_collate
from .models import TemporalNet, apply_temporal_correction
from .fusion import FusionNet, build_fusion_features, precision_fuse
from .io import write_product


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(batch: dict[str, torch.Tensor], temporal_net: TemporalNet, fusion_net: FusionNet, cfg: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prediction = temporal_net(batch, cfg)
    temporal = apply_temporal_correction(batch, prediction, cfg)
    features = build_fusion_features(batch, temporal)
    fusion = fusion_net(features, temporal["source_mask"])
    product = precision_fuse(batch, temporal, fusion)
    return temporal, fusion, product


def _permutation_difference(batch: dict[str, torch.Tensor], temporal: dict[str, torch.Tensor], fusion_net: FusionNet, cfg: dict[str, Any], product: dict[str, torch.Tensor]) -> torch.Tensor:
    nsource = temporal["source_mask"].shape[1]
    if nsource < 2:
        return product["fused_r"].sum() * 0.0
    perm = torch.arange(nsource - 1, -1, -1, device=temporal["source_mask"].device)
    perm_temporal = {key: (value[:, perm] if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == nsource else value) for key, value in temporal.items()}
    perm_batch = dict(batch)
    for key in ("features", "tau_hours", "source_coverages"):
        perm_batch[key] = batch[key][:, perm]
    perm_features = build_fusion_features(perm_batch, perm_temporal)
    perm_fusion = fusion_net(perm_features, perm_temporal["source_mask"])
    perm_product = precision_fuse(perm_batch, perm_temporal, perm_fusion)
    mask = batch["hidden_mask"].bool() & torch.isfinite(product["fused_r"]) & torch.isfinite(perm_product["fused_r"])
    if not torch.any(mask):
        return product["fused_r"].sum() * 0.0
    return ((product["fused_r"] - perm_product["fused_r"]) ** 2 + (product["fused_t"] - perm_product["fused_t"]) ** 2)[mask].mean()


def _method_metrics(pred_r: torch.Tensor, pred_t: torch.Tensor, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    target_r = batch["reference_target_r"]
    target_t = batch["reference_target_t"]
    finite = mask & torch.isfinite(pred_r) & torch.isfinite(pred_t)
    n = int(finite.sum().detach().cpu())
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "speed_rmse": float("nan"), "direction_error_deg": float("nan"), "n_points": 0, "coverage": 0.0}
    er, et = pred_r - target_r, pred_t - target_t
    vec = torch.sqrt(er.square() + et.square())
    speed = torch.sqrt(pred_r.square() + pred_t.square()) - torch.sqrt(target_r.square() + target_t.square())
    cu, cv = batch["cos_theta"], batch["sin_theta"]
    pu, pv = pred_r * cu - pred_t * cv, pred_r * cv + pred_t * cu
    tu, tv = target_r * cu - target_t * cv, target_r * cv + target_t * cu
    denom = (torch.sqrt(pu.square() + pv.square()) * torch.sqrt(tu.square() + tv.square())).clamp_min(1.0e-6)
    angle = torch.rad2deg(torch.acos(torch.clamp((pu * tu + pv * tv) / denom, -1.0, 1.0)))
    hidden = batch["hidden_mask"]
    return {"rmse": float(torch.sqrt(torch.mean(vec[finite].square())).detach().cpu()), "mae": float(torch.mean(vec[finite]).detach().cpu()), "speed_rmse": float(torch.sqrt(torch.mean(speed[finite].square())).detach().cpu()), "direction_error_deg": float(torch.mean(angle[finite]).detach().cpu()), "n_points": n, "coverage": float(n / max(int(hidden.sum().detach().cpu()), 1))}


def _raw_mean(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = batch["source_support"].bool()
    count = support.sum(dim=1)
    raw_r = (batch["source_radial"] * support).sum(dim=1) / count.clamp_min(1) 
    raw_t = (batch["source_tangential"] * support).sum(dim=1) / count.clamp_min(1)
    valid = count > 0
    raw_r = torch.where(valid, raw_r, torch.full_like(raw_r, float("nan")))
    raw_t = torch.where(valid, raw_t, torch.full_like(raw_t, float("nan")))
    ref = batch["reference_mask"].bool()
    raw_r = torch.where(ref, batch["reference_context_r"], raw_r)
    raw_t = torch.where(ref, batch["reference_context_t"], raw_t)
    return raw_r, raw_t, count


def _equal_mean(temporal: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = temporal["source_mask"].bool()
    count = support.sum(dim=1)
    r = (temporal["corrected_r"] * support).sum(dim=1) / count.clamp_min(1)
    t = (temporal["corrected_t"] * support).sum(dim=1) / count.clamp_min(1)
    valid = count > 0
    r = torch.where(valid, r, torch.full_like(r, float("nan")))
    t = torch.where(valid, t, torch.full_like(t, float("nan")))
    ref = batch["reference_mask"].bool()
    r, t = torch.where(ref, batch["reference_context_r"], r), torch.where(ref, batch["reference_context_t"], t)
    return r, t, count


def evaluate_loader(temporal_net: TemporalNet, fusion_net: FusionNet, loader: DataLoader, cfg: dict[str, Any], device: torch.device, split: str, fold: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    temporal_net.eval(); fusion_net.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_cpu in loader:
            batch = _move(batch_cpu, device)
            temporal, fusion, product = _forward(batch, temporal_net, fusion_net, cfg)
            raw_r, raw_t, raw_count = _raw_mean(batch)
            equal_r, equal_t, equal_count = _equal_mean(temporal, batch)
            methods = {"raw_equal": (raw_r, raw_t, raw_count > 0), "harmonized_equal": (equal_r, equal_t, equal_count > 0), "reference_time": (product["fused_r"], product["fused_t"], torch.isfinite(product["fused_r"]) & torch.isfinite(product["fused_t"]))}
            common = batch["hidden_mask"].bool() & methods["raw_equal"][2] & methods["harmonized_equal"][2] & methods["reference_time"][2]
            for method, (r, t, available) in methods.items():
                metrics = _method_metrics(r, t, batch, batch["hidden_mask"].bool() & available)
                common_metrics = _method_metrics(r, t, batch, common)
                rows.append({"split": split, "fold": fold, "mission_id": batch["mission_id"][0], "storm_key": batch["storm_key"][0], "pseudo_reference_swath_id": batch["pseudo_reference_swath_id"][0], "method": method, "rmse": metrics["rmse"], "mae": metrics["mae"], "speed_rmse": metrics["speed_rmse"], "direction_error_deg": metrics["direction_error_deg"], "coverage": metrics["coverage"], "n_points": metrics["n_points"], "common_rmse": common_metrics["rmse"], "common_points": common_metrics["n_points"]})
            for source_index, source_id in enumerate(batch["source_swath_ids"][0]):
                raw_mask = batch["hidden_mask"].bool() & batch["source_support"][:, source_index].bool()
                aligned_mask = batch["hidden_mask"].bool() & temporal["source_mask"][:, source_index].bool()
                for method, r, t, mask in (("raw_source", batch["source_radial"][:, source_index], batch["source_tangential"][:, source_index], raw_mask), ("corrected_source", temporal["corrected_r"][:, source_index], temporal["corrected_t"][:, source_index], aligned_mask)):
                    metrics = _method_metrics(r, t, batch, mask)
                    rows.append({"split": split, "fold": fold, "mission_id": batch["mission_id"][0], "storm_key": batch["storm_key"][0], "source_swath_id": source_id, "pseudo_reference_swath_id": batch["pseudo_reference_swath_id"][0], "method": method, "rmse": metrics["rmse"], "mae": metrics["mae"], "speed_rmse": metrics["speed_rmse"], "direction_error_deg": metrics["direction_error_deg"], "coverage": metrics["coverage"], "n_points": metrics["n_points"], "tau_hours": float(batch["tau_hours"][0, source_index].detach().cpu())})
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    if not frame.empty:
        for method, value in frame[frame["method"].isin(["raw_equal", "harmonized_equal", "reference_time"])].groupby("method")["rmse"].mean().items():
            summary[f"{method}_rmse"] = float(value)
        summary["n_pairs"] = float(frame["mission_id"].nunique())
        summary["n_storms"] = float(frame["storm_key"].nunique())
    return summary, rows


def _load_models(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[TemporalNet, FusionNet]:
    temporal = TemporalNet(cfg).to(device)
    fusion = FusionNet(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    temporal.load_state_dict(state["temporal"]); fusion.load_state_dict(state["fusion"])
    temporal.eval(); fusion.eval()
    return temporal, fusion


def _write_archive_product(cfg: dict[str, Any], output: Path, batch_cpu: dict[str, Any], temporal: dict[str, torch.Tensor], product: dict[str, torch.Tensor], checkpoint: Path) -> Path:
    batch = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch_cpu.items()}
    temp = {key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.shape[0] == 1 else value.detach().cpu().numpy() if torch.is_tensor(value) else value for key, value in temporal.items()}
    prod = {key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.shape[0] == 1 else value.detach().cpu().numpy() if torch.is_tensor(value) else value for key, value in product.items()}
                                                                               
                                                                      
    fused_r, fused_t = prod["fused_r"], prod["fused_t"]
    cos_t = batch["cos_theta"].numpy()[0]; sin_t = batch["sin_theta"].numpy()[0]
    fused_u, fused_v = fused_r * cos_t - fused_t * sin_t, fused_r * sin_t + fused_t * cos_t
    has_candidate = np.asarray(prod["has_candidate"], dtype=bool)
    ref_r, ref_t = batch["reference_target_r"].numpy()[0], batch["reference_target_t"].numpy()[0]
    ref_mask = batch["reference_mask"].numpy()[0].astype(bool)
    reference_u, reference_v = ref_r * cos_t - ref_t * sin_t, ref_r * sin_t + ref_t * cos_t
    reference_u = np.where(ref_mask, reference_u, np.nan).astype(np.float32); reference_v = np.where(ref_mask, reference_v, np.nan).astype(np.float32)
    fused_u = np.where(ref_mask | has_candidate, fused_u, np.nan).astype(np.float32)
    fused_v = np.where(ref_mask | has_candidate, fused_v, np.nan).astype(np.float32)
    var_r, var_t, cov = prod["variance_r"], prod["variance_t"], prod["covariance_rt"]
    var_u = var_r * cos_t ** 2 + var_t * sin_t ** 2 - 2.0 * cov * sin_t * cos_t
    var_v = var_r * sin_t ** 2 + var_t * cos_t ** 2 + 2.0 * cov * sin_t * cos_t
    payload = {
        "fused_u": fused_u, "fused_v": fused_v, "uncertainty_u": np.sqrt(np.maximum(var_u, 0.0)), "uncertainty_v": np.sqrt(np.maximum(var_v, 0.0)), "uncertainty_speed": np.sqrt(np.maximum((var_r + var_t) / 2.0, 0.0)),
        "reference_u": reference_u, "reference_v": reference_v, "reference_mask": ref_mask,
        "candidate_support_count": prod["support_count"], "effective_source_count": prod["effective_source_count"], "provenance": prod["provenance"], "weights": prod["weights"],
        "candidate_support": temp["source_mask"], "candidate_raw_u": np.where(batch["source_support"].numpy()[0].astype(bool), batch["source_u"].numpy()[0], np.nan), "candidate_raw_v": np.where(batch["source_support"].numpy()[0].astype(bool), batch["source_v"].numpy()[0], np.nan),
        "candidate_corrected_u": temp["corrected_u"], "candidate_corrected_v": temp["corrected_v"],
        "increment_b_r": temp["increment_b_r"], "increment_b_t": temp["increment_b_t"], "increment_l_r": temp["increment_l_r"], "increment_l_t": temp["increment_l_t"], "increment_s_r": temp["increment_s_r"], "increment_s_t": temp["increment_s_t"],
        "covariance_rt": cov, "tau_hours": batch["tau_hours"].numpy()[0], "source_time_hours": -batch["tau_hours"].numpy()[0], "source_swath_ids": batch["source_swath_ids"][0], "reference_swath_id": batch["pseudo_reference_swath_id"][0],
    }
                                                                                
    payload["sigma_r"] = temp["sigma_r_eff"][:, None, None, None] * np.ones_like(temp["corrected_r"])
    payload["sigma_t"] = temp["sigma_t_eff"][:, None, None, None] * np.ones_like(temp["corrected_t"])
    payload["rho"] = temp["rho"][:, None, None, None] * np.ones_like(temp["corrected_r"])
    payload["quality"] = temp["quality"][:, None, None, None] * np.ones_like(temp["corrected_r"])
    mission_id = str(batch["mission_id"][0])
    path = output / "products" / f"MLNEW_FUSED_{mission_id}.nc"
    metadata = {"model_name": "TC-RADAR reference-time fusion", "model_version": cfg["project"]["version"], "reference_swath": batch["pseudo_reference_swath_id"][0], "source_swaths": batch["source_swath_ids"][0], "checkpoint": str(checkpoint.resolve()), "config": str(Path(cfg["_config_path"]).resolve()), "split": "archive", "reference_time_convention": "source_time_hours are relative to reference midpoint"}
    return write_product(path, payload, {"x": batch["x_km"].numpy()[0], "y": batch["y_km"].numpy()[0], "z": batch["z_km"].numpy()[0]}, metadata)

