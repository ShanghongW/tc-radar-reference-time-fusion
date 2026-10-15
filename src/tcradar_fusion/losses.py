from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not torch.any(mask):
        return value.sum() * 0.0
    return value[mask].mean()


def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(error, torch.zeros_like(error), delta=float(delta), reduction="none")


def _candidate_nll(error_r: torch.Tensor, error_t: torch.Tensor, sigma_r: torch.Tensor, sigma_t: torch.Tensor, rho: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sr = sigma_r.clamp_min(0.05)
    st = sigma_t.clamp_min(0.05)
    rr = rho.clamp(-0.97, 0.97)
    cov = rr * sr * st
    det = (sr.square() * st.square() - cov.square()).clamp_min(1.0e-6)
    quad = (st.square() * error_r.square() - 2.0 * cov * error_r * error_t + sr.square() * error_t.square()) / det
    nll = 0.5 * (quad + torch.log(det))
    return _masked_mean(nll, mask)


def temporal_loss(batch: dict[str, torch.Tensor], temporal: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    target_r = batch["reference_target_r"][:, None]
    target_t = batch["reference_target_t"][:, None]
    hidden = batch["hidden_mask"][:, None].bool()
    support = temporal["source_mask"].bool()
    mask = hidden & support
    er = temporal["corrected_r"] - target_r
    et = temporal["corrected_t"] - target_t
    delta = float(cfg["loss"]["huber_transition_ms"])
    vector = _masked_mean(_huber(er, delta) + _huber(et, delta), mask)
    nll = _candidate_nll(er, et, temporal["sigma_r_eff"][:, :, None, None, None], temporal["sigma_t_eff"][:, :, None, None, None], temporal["rho"][:, :, None, None, None], mask)
                                                                         
                                                                     
    target_small_r = target_r - batch["source_background_r"][:, :1]                                              
    target_small_t = target_t - batch["source_background_t"][:, :1]
    multi = _masked_mean(_huber(er, delta) + _huber(et, delta), mask) + 0.05 * _masked_mean(torch.abs(target_small_r * 0.0 + er) + torch.abs(target_small_t * 0.0 + et), mask)
    zero = batch["tau_hours"].abs() < 1.0e-8
    identity_mask = zero[:, :, None, None, None] & support
    identity = _masked_mean(torch.abs(er) + torch.abs(et), identity_mask)
    tendency = temporal["tendency"].square().mean() + temporal["kappa"].square().mean()
    smooth = torch.tensor(0.0, device=er.device, dtype=er.dtype)
    for field in (temporal["tendency"], temporal["kappa"]):
        smooth = smooth + field.diff(dim=-1).square().mean() + field.diff(dim=-2).square().mean()
    zero_overlap = torch.tensor(0.0, device=er.device, dtype=er.dtype)
    overlap = batch["features"][:, :, 21] < 1.0e-8
    zero_overlap = _masked_mean(_huber(er, delta) + _huber(et, delta), mask & overlap)
    consistency = torch.tensor(0.0, device=er.device, dtype=er.dtype)
    if temporal["corrected_r"].shape[1] > 1:
        for i in range(temporal["corrected_r"].shape[1]):
            for j in range(i + 1, temporal["corrected_r"].shape[1]):
                both = hidden[:, 0] & support[:, i] & support[:, j]
                consistency = consistency + _masked_mean(_huber(temporal["corrected_r"][:, i] - temporal["corrected_r"][:, j], delta) + _huber(temporal["corrected_t"][:, i] - temporal["corrected_t"][:, j], delta), both)
    weights = cfg["loss"]
    total = (
        vector + float(weights["temporal_nll"]) * nll + float(weights["temporal_multiscale"]) * multi
        + float(weights["temporal_identity"]) * identity + float(weights["temporal_tendency"]) * tendency
        + float(weights["temporal_smooth"]) * smooth + float(weights["temporal_zero_overlap"]) * zero_overlap
        + float(weights["temporal_consistency"]) * consistency
    )
    return total, {"vector": float(vector.detach()), "nll": float(nll.detach()), "multi": float(multi.detach()), "identity": float(identity.detach()), "tendency": float(tendency.detach()), "smooth": float(smooth.detach()), "zero_overlap": float(zero_overlap.detach()), "consistency": float(consistency.detach())}


def fusion_loss(batch: dict[str, torch.Tensor], temporal: dict[str, torch.Tensor], fusion: dict[str, torch.Tensor], product: dict[str, torch.Tensor], cfg: dict[str, Any], permutation_difference: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    hidden = batch["hidden_mask"].bool()
    pred_r, pred_t = product["fused_r"], product["fused_t"]
    target_r, target_t = batch["reference_target_r"], batch["reference_target_t"]
    final_mask = hidden & torch.isfinite(pred_r) & torch.isfinite(pred_t)
    delta = float(cfg["loss"]["huber_transition_ms"])
    final = _masked_mean(_huber(pred_r - target_r, delta) + _huber(pred_t - target_t, delta), final_mask)
    nll = _candidate_nll(pred_r - target_r, pred_t - target_t, torch.sqrt(product["variance_r"].clamp_min(1.0e-5)), torch.sqrt(product["variance_t"].clamp_min(1.0e-5)), product["covariance_rt"] / torch.sqrt(product["variance_r"].clamp_min(1.0e-5) * product["variance_t"].clamp_min(1.0e-5)), final_mask)
    cand_mask = hidden[:, None] & temporal["source_mask"].bool()
    er = temporal["corrected_r"] - target_r[:, None]
    et = temporal["corrected_t"] - target_t[:, None]
    cand_nll = _candidate_nll(er, et, temporal["sigma_r_eff"][:, :, None, None, None] * fusion["scale_r"], temporal["sigma_t_eff"][:, :, None, None, None] * fusion["scale_t"], temporal["rho"][:, :, None, None, None] + fusion["rho_delta"], cand_mask)
    actual = torch.sqrt(er.square() + et.square() + 1.0e-6)
    precision = fusion["reliability"] / (temporal["sigma_r_eff"][:, :, None, None, None] * temporal["sigma_t_eff"][:, :, None, None, None] * fusion["scale_r"] * fusion["scale_t"] + 1.0e-5)
    rank_terms = []
    for i in range(actual.shape[1]):
        for j in range(i + 1, actual.shape[1]):
            both = cand_mask[:, i] & cand_mask[:, j]
            better_i = actual[:, i] < actual[:, j]
            better_j = ~better_i
            rank_terms.append(_masked_mean(torch.relu(0.01 - (precision[:, i] - precision[:, j])), both & better_i))
            rank_terms.append(_masked_mean(torch.relu(0.01 - (precision[:, j] - precision[:, i])), both & better_j))
    rank = torch.stack(rank_terms).mean() if rank_terms else final * 0.0
    permutation = permutation_difference if permutation_difference is not None else final * 0.0
    weight_smooth = product["weights"].diff(dim=-1).abs().mean() + product["weights"].diff(dim=-2).abs().mean()
    weights = cfg["loss"]
    total = final + float(weights["fusion_nll"]) * nll + float(weights["fusion_candidate_nll"]) * cand_nll + float(weights["fusion_rank"]) * rank + float(weights["fusion_permutation"]) * permutation + float(weights["fusion_weight_smooth"]) * weight_smooth
    return total, {"final": float(final.detach()), "nll": float(nll.detach()), "candidate_nll": float(cand_nll.detach()), "rank": float(rank.detach()), "permutation": float(permutation.detach()), "weight_smooth": float(weight_smooth.detach())}

