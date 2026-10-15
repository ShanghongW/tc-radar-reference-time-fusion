from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TemporalNet(nn.Module):
    """Pointwise temporal tendency network conditioned on height and storm geometry.

    The spatial grid is never resampled.  A shared pointwise encoder predicts
    bounded radial/tangential tendencies for the fixed B/L/S components, while
    a pooled metadata head predicts candidate covariance and quality.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        tcfg = cfg["temporal_model"]
        channels = int(tcfg["input_channels"])
        hidden = int(tcfg["hidden_channels"])
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.field_head = nn.Conv3d(hidden, 9, kernel_size=1)
        self.global_head = nn.Sequential(
            nn.Linear(hidden + 4, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, 5)
        )
        self.max_tendency = float(tcfg["max_tendency_ms_per_hour"])
        self.max_small_tendency = float(tcfg["max_small_tendency_ms_per_hour"])
        self.max_kappa = float(tcfg["max_kappa_per_hour"])
        self.min_sigma = float(tcfg["min_sigma_ms"])
        self.max_sigma = float(tcfg["max_sigma_ms"])

    def forward(self, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = torch.nan_to_num(batch["features"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        bsz, nsource, channels, nz, ny, nx = features.shape
        flat = features.reshape(bsz * nsource, channels, nz, ny, nx)
        encoded = self.encoder(flat)
        field = self.field_head(encoded)
        raw_tendency = torch.tanh(field[:, :6]).reshape(bsz, nsource, 6, nz, ny, nx)
        tendency = torch.cat(
            [raw_tendency[:, :, :4] * self.max_tendency, raw_tendency[:, :, 4:6] * self.max_small_tendency],
            dim=2,
        )
        kappa = (torch.tanh(field[:, 6:9]) * self.max_kappa).reshape(bsz, nsource, 3, nz, ny, nx)
        pooled = encoded.mean(dim=(-3, -2, -1))
        tau = batch["tau_hours"].float()
        meta = torch.stack(
            [tau / 4.0, tau.abs() / 4.0, torch.sign(tau), batch["source_coverages"].float()], dim=-1
        ).reshape(bsz * nsource, 4)
        global_out = self.global_head(torch.cat([pooled, meta], dim=-1)).reshape(bsz, nsource, 5)
                                                                      
                                                                              
        l11 = F.softplus(global_out[..., 0]) + self.min_sigma
        l21 = 0.75 * torch.tanh(global_out[..., 1]) * self.max_sigma
        l22 = F.softplus(global_out[..., 2]) + self.min_sigma
        sigma_r = l11.clamp(max=self.max_sigma)
        sigma_t = torch.sqrt(l21.square() + l22.square() + 1.0e-6).clamp(max=self.max_sigma)
        rho = (l21 / sigma_t).clamp(-0.95, 0.95)
        quality = torch.sigmoid(global_out[..., 3])
                                                                              
                                                                          
        temporal_scale = 0.5 + torch.sigmoid(global_out[..., 4])
        return {
            "tendency": tendency,
            "kappa": kappa,
            "sigma_r": sigma_r,
            "sigma_t": sigma_t,
            "rho": rho,
            "quality": quality,
            "temporal_scale": temporal_scale,
        }


def apply_temporal_correction(batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Apply signed-time B/L/S correction at the original source cells."""
    tau = batch["tau_hours"].float()
    signed = tau[:, :, None, None, None, None]
    absolute = tau.abs()[:, :, None, None, None, None]
    # Ordered as [T_B, T_L, T_S] in hours; the supplied configuration sets all three to 12 h.
    scales = torch.as_tensor(cfg["temporal_model"]["time_scales_hours"], dtype=tau.dtype, device=tau.device).reshape(1, 1, 3, 1, 1, 1)
    decay = torch.exp(-absolute / scales)
    tendency = prediction["tendency"]
    kappa = prediction["kappa"]
                                                                            
                                                                             
    growth = torch.exp(torch.clamp(signed * kappa, min=-0.75, max=0.75))
    b_r = batch["source_background_r"]
    b_t = batch["source_background_t"]
    l_r = batch["source_large_r"]
    l_t = batch["source_large_t"]
    s_r = batch["source_small_r"]
    s_t = batch["source_small_t"]
    inc_b_r = signed[:, :, 0] * decay[:, :, 0] * tendency[:, :, 0] 
    inc_b_t = signed[:, :, 0] * decay[:, :, 0] * tendency[:, :, 1]
    inc_l_r = signed[:, :, 0] * decay[:, :, 1] * growth[:, :, 1] * tendency[:, :, 2]
    inc_l_t = signed[:, :, 0] * decay[:, :, 1] * growth[:, :, 1] * tendency[:, :, 3]
    inc_s_r = signed[:, :, 0] * decay[:, :, 2] * growth[:, :, 2] * prediction["temporal_scale"][:, :, None, None, None] * tendency[:, :, 4]
    inc_s_t = signed[:, :, 0] * decay[:, :, 2] * growth[:, :, 2] * prediction["temporal_scale"][:, :, None, None, None] * tendency[:, :, 5]
    corrected_r = b_r + l_r + s_r + inc_b_r + inc_l_r + inc_s_r
    corrected_t = b_t + l_t + s_t + inc_b_t + inc_l_t + inc_s_t
    zero = tau.abs() < 1.0e-8
    corrected_r = torch.where(zero[:, :, None, None, None], batch["source_radial"], corrected_r)
    corrected_t = torch.where(zero[:, :, None, None, None], batch["source_tangential"], corrected_t)
    support = batch["source_support"].float()
    corrected_r = corrected_r * support
    corrected_t = corrected_t * support
    cos_t = batch["cos_theta"][:, None]
    sin_t = batch["sin_theta"][:, None]
    corrected_u = corrected_r * cos_t - corrected_t * sin_t
    corrected_v = corrected_r * sin_t + corrected_t * cos_t
    sigma_r = prediction["sigma_r"] * (1.0 + 0.10 * tau.abs())
    sigma_t = prediction["sigma_t"] * (1.0 + 0.10 * tau.abs())
    return {
        **prediction,
        "corrected_r": corrected_r,
        "corrected_t": corrected_t,
        "corrected_u": corrected_u,
        "corrected_v": corrected_v,
        "source_mask": support,
        "increment_b_r": inc_b_r,
        "increment_b_t": inc_b_t,
        "increment_l_r": inc_l_r,
        "increment_l_t": inc_l_t,
        "increment_s_r": inc_s_r,
        "increment_s_t": inc_s_t,
        "sigma_r_eff": sigma_r,
        "sigma_t_eff": sigma_t,
    }
