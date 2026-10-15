from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _safe_inverse_2x2(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    det = (a * d - b * b).clamp_min(1.0e-8)
    return d / det, -b / det, a / det


class FusionNet(nn.Module):
    """Small Deep-Sets reliability head; it cannot directly generate wind."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        fcfg = cfg["fusion_model"]
        dim = int(fcfg["candidate_channels"])
        hidden = int(fcfg["hidden_channels"])
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.psi = nn.Sequential(nn.Linear(dim + 3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.min_q = float(fcfg["min_reliability"])
        self.max_q = float(fcfg["max_reliability"])
        self.min_scale = float(fcfg["min_sigma_scale"])
        self.max_scale = float(fcfg["max_sigma_scale"])

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, nsource, dim, nz, ny, nx = features.shape
        x = torch.nan_to_num(features.permute(0, 1, 3, 4, 5, 2), nan=0.0, posinf=0.0, neginf=0.0)
        flat = x.reshape(-1, dim)
        h = self.phi(flat).reshape(bsz, nsource, nz, ny, nx, -1)
        m = valid.permute(0, 1, 2, 3, 4).float().unsqueeze(-1)
        count = m.sum(dim=1).clamp_min(1.0)
        h_mean = (h * m).sum(dim=1) / count
        h_max = h.masked_fill(m <= 0.0, -1.0e4).max(dim=1).values
        h_std = torch.sqrt(((h - h_mean[:, None]) ** 2 * m).sum(dim=1) / count + 1.0e-6)
        set_stats = torch.cat([h_mean, h_max, h_std], dim=-1)
        set_broadcast = set_stats[:, None].expand(-1, nsource, -1, -1, -1, -1)
        psi_in = torch.cat([x, set_broadcast], dim=-1).reshape(-1, dim + 3 * h.shape[-1])
        out = self.psi(psi_in).reshape(bsz, nsource, nz, ny, nx, 4).permute(0, 1, 5, 2, 3, 4)
        q = self.min_q + (self.max_q - self.min_q) * torch.sigmoid(out[:, :, 0])
        scale_r = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(out[:, :, 1])
        scale_t = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(out[:, :, 2])
        rho_delta = 0.25 * torch.tanh(out[:, :, 3])
        return {"reliability": q, "scale_r": scale_r, "scale_t": scale_t, "rho_delta": rho_delta}


def build_fusion_features(batch: dict[str, torch.Tensor], temporal: dict[str, torch.Tensor]) -> torch.Tensor:
    support = temporal["source_mask"]
    inc_b = torch.sqrt(temporal["increment_b_r"] ** 2 + temporal["increment_b_t"] ** 2)
    inc_l = torch.sqrt(temporal["increment_l_r"] ** 2 + temporal["increment_l_t"] ** 2)
    inc_s = torch.sqrt(temporal["increment_s_r"] ** 2 + temporal["increment_s_t"] ** 2)
    overlap = batch["features"][:, :, 21]
    coverage = batch["features"][:, :, 19]
    rnorm = batch["radius_grid"][:, None] / 150.0
    z_norm = batch["z_norm"][:, None]
    return torch.stack([
        temporal["corrected_r"], temporal["corrected_t"], temporal["corrected_u"], temporal["corrected_v"],
        temporal["sigma_r_eff"][:, :, None, None, None].expand_as(temporal["corrected_r"]),
        temporal["sigma_t_eff"][:, :, None, None, None].expand_as(temporal["corrected_t"]),
        temporal["rho"][:, :, None, None, None].expand_as(temporal["corrected_r"]),
        temporal["quality"][:, :, None, None, None].expand_as(temporal["corrected_r"]),
        batch["tau_hours"][:, :, None, None, None].expand_as(temporal["corrected_r"]) / 4.0,
        batch["tau_hours"].abs()[:, :, None, None, None].expand_as(temporal["corrected_r"]) / 4.0,
        inc_b, inc_l, inc_s, support, overlap, coverage,
        rnorm.expand_as(temporal["corrected_r"]), z_norm.expand_as(temporal["corrected_r"]),
    ], dim=2)


def precision_fuse(batch: dict[str, torch.Tensor], temporal: dict[str, torch.Tensor], fusion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = temporal["source_mask"]
    sr = (temporal["sigma_r_eff"][:, :, None, None, None] * fusion["scale_r"]).clamp_min(0.05)
    st = (temporal["sigma_t_eff"][:, :, None, None, None] * fusion["scale_t"]).clamp_min(0.05)
    rho = torch.clamp(temporal["rho"][:, :, None, None, None] + fusion["rho_delta"], -0.97, 0.97)
    vr, vt = sr.square(), st.square()
    cov = rho * sr * st
    prr, prt, ptt = _safe_inverse_2x2(vr, cov, vt)
    q = fusion["reliability"] * valid
    prr, prt, ptt = prr * q, prt * q, ptt * q
    sum_prr, sum_prt, sum_ptt = prr.sum(dim=1), prt.sum(dim=1), ptt.sum(dim=1)
    det_sum = (sum_prr * sum_ptt - sum_prt.square()).clamp_min(1.0e-8)
    rhs_r = (prr * temporal["corrected_r"] + prt * temporal["corrected_t"]).sum(dim=1)
    rhs_t = (prt * temporal["corrected_r"] + ptt * temporal["corrected_t"]).sum(dim=1)
    fused_r = (sum_ptt * rhs_r - sum_prt * rhs_t) / det_sum
    fused_t = (-sum_prt * rhs_r + sum_prr * rhs_t) / det_sum
    count = valid.sum(dim=1)
    has = count > 0
                                                                        
                                                                              
                                                                          
                                                                           
    fused_r = torch.where(has, fused_r, torch.zeros_like(fused_r))
    fused_t = torch.where(has, fused_t, torch.zeros_like(fused_t))
    var_r = sum_ptt / det_sum
    var_t = sum_prr / det_sum
    cov_f = -sum_prt / det_sum
    scalar_precision = (prr + ptt).clamp_min(0.0)
    scalar_sum = scalar_precision.sum(dim=1).clamp_min(1.0e-8)
    weights = scalar_precision / scalar_sum[:, None]
    eff_n = scalar_precision.sum(dim=1).square() / scalar_precision.square().sum(dim=1).clamp_min(1.0e-8)
                                                                            
    ref = batch["reference_mask"].bool()
    fused_r = torch.where(ref, batch["reference_context_r"], fused_r)
    fused_t = torch.where(ref, batch["reference_context_t"], fused_t)
    var_r = torch.where(ref, torch.zeros_like(var_r), var_r)
    var_t = torch.where(ref, torch.zeros_like(var_t), var_t)
    cov_f = torch.where(ref, torch.zeros_like(cov_f), cov_f)
    provenance = torch.where(ref, torch.ones_like(count, dtype=torch.int8), torch.where(count == 0, torch.zeros_like(count, dtype=torch.int8), torch.where(count == 1, torch.full_like(count, 2, dtype=torch.int8), torch.full_like(count, 3, dtype=torch.int8))))
    return {
        "fused_r": fused_r, "fused_t": fused_t, "variance_r": var_r, "variance_t": var_t, "covariance_rt": cov_f,
        "support_count": count, "has_candidate": has, "effective_source_count": eff_n, "weights": weights, "provenance": provenance,
    }
