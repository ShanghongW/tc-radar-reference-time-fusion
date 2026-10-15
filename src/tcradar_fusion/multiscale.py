"""Fixed multiscale decomposition of each observed source swath.

The operators in this module preserve grid locations.  They only transform
wind-vector components at the same grid point and never move support.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import polar_geometry, radial_tangential_to_uv, uv_to_radial_tangential


def _robust_profile(
    field: np.ndarray,
    support: np.ndarray,
    radius: np.ndarray,
    radial_bin_km: float,
    radius_max_km: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return robust annular profile, valid fraction, and MAD uncertainty."""
    nz = field.shape[0]
    edges = np.arange(0.0, radius_max_km + radial_bin_km * 1.001, radial_bin_km)
    if edges.size < 2:
        edges = np.asarray([0.0, radial_bin_km])
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = np.clip(np.digitize(radius, edges) - 1, 0, len(centers) - 1)
    profile = np.full((nz, len(centers)), np.nan, dtype=np.float32)
    fraction = np.zeros_like(profile)
    uncertainty = np.full_like(profile, np.nan)
    for iz in range(nz):
        inside = radius <= radius_max_km
        total = np.bincount(bins[inside].ravel(), minlength=len(centers)).astype(float)
        valid = support[iz] & np.isfinite(field[iz]) & inside
        for ir in range(len(centers)):
            values = field[iz][valid & (bins == ir)]
            if values.size >= int(min_points):
                med = float(np.nanmedian(values))
                mad = float(np.nanmedian(np.abs(values - med))) * 1.4826
                profile[iz, ir] = med
                uncertainty[iz, ir] = mad
                fraction[iz, ir] = float(values.size / max(total[ir], 1.0))
        known = np.isfinite(profile[iz])
        if known.any():
            if known.sum() == 1:
                profile[iz, ~known] = profile[iz, known][0]
                uncertainty[iz, ~known] = uncertainty[iz, known][0]
            else:
                profile[iz, ~known] = np.interp(centers[~known], centers[known], profile[iz, known])
                uncertainty[iz, ~known] = np.interp(centers[~known], centers[known], uncertainty[iz, known])
                                                                   
            fraction[iz, ~known] = 0.0
        else:
            profile[iz] = 0.0
            uncertainty[iz] = 0.0
    return profile, fraction, uncertainty


def _expand_profile(profile: np.ndarray, radius: np.ndarray, radial_bin_km: float) -> np.ndarray:
    centers = (np.arange(profile.shape[1], dtype=np.float32) + 0.5) * float(radial_bin_km)
    bins = np.clip(np.digitize(radius, np.r_[0.0, centers + radial_bin_km / 2.0]) - 1, 0, len(centers) - 1)
    return profile[:, bins].astype(np.float32)


def _normalised_gaussian(field: np.ndarray, support: np.ndarray, sigma_pixels: float, threshold: float) -> np.ndarray:
    out = np.zeros_like(field, dtype=np.float32)
    for iz in range(field.shape[0]):
        m = support[iz].astype(np.float32)
        numerator = gaussian_filter(np.nan_to_num(field[iz], nan=0.0) * m, sigma=sigma_pixels, mode="constant", cval=0.0)
        denominator = gaussian_filter(m, sigma=sigma_pixels, mode="constant", cval=0.0)
        ok = denominator >= float(threshold)
        out[iz, ok] = numerator[ok] / np.maximum(denominator[ok], 1.0e-8)
    return out


def decompose_source(
    u: np.ndarray,
    v: np.ndarray,
    support: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    """Decompose one source into B + L + S in radial/tangential components."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    radius, _ = polar_geometry(x_km, y_km)
    radial, tangential = uv_to_radial_tangential(u, v, x_km, y_km)
    radial[~support] = np.nan
    tangential[~support] = np.nan
    rmax = float(cfg["radius_max_km"])
    rb = float(cfg["radial_bin_km"])
    min_points = int(cfg.get("min_annular_points", 8))
    br_profile, br_fraction, br_unc = _robust_profile(radial, support, radius, rb, rmax, min_points)
    bt_profile, bt_fraction, bt_unc = _robust_profile(tangential, support, radius, rb, rmax, min_points)
    background_r = _expand_profile(br_profile, radius, rb)
    background_t = _expand_profile(bt_profile, radius, rb)
    perturb_r = np.where(support, np.nan_to_num(radial, nan=0.0) - background_r, 0.0).astype(np.float32)
    perturb_t = np.where(support, np.nan_to_num(tangential, nan=0.0) - background_t, 0.0).astype(np.float32)
    dx = float(np.nanmedian(np.diff(np.asarray(x_km, dtype=float))))
    dy = float(np.nanmedian(np.diff(np.asarray(y_km, dtype=float))))
    sigma_km = float(cfg.get("large_scale_sigma_km", 12.0))
    sigma_px = max(0.5, sigma_km / max(np.sqrt(abs(dx * dy)), 1.0e-6))
    large_r = _normalised_gaussian(perturb_r, support, sigma_px, float(cfg.get("gaussian_mask_threshold", 0.25)))
    large_t = _normalised_gaussian(perturb_t, support, sigma_px, float(cfg.get("gaussian_mask_threshold", 0.25)))
                                                                             
                                                                            
                                                                             
    large_r[~support] = 0.0
    large_t[~support] = 0.0
    small_r = np.where(support, perturb_r - large_r, 0.0).astype(np.float32)
    small_t = np.where(support, perturb_t - large_t, 0.0).astype(np.float32)
    recon_r = background_r + large_r + small_r
    recon_t = background_t + large_t + small_t
    reconstruction_error = float(np.nanmax(np.abs((recon_r - radial)[support])) if np.any(support) else 0.0)
    return {
        "source_radial": np.nan_to_num(radial, nan=0.0).astype(np.float32),
        "source_tangential": np.nan_to_num(tangential, nan=0.0).astype(np.float32),
        "background_r": background_r.astype(np.float32),
        "background_t": background_t.astype(np.float32),
        "large_r": large_r.astype(np.float32),
        "large_t": large_t.astype(np.float32),
        "small_r": small_r.astype(np.float32),
        "small_t": small_t.astype(np.float32),
        "annular_valid_fraction": np.minimum(br_fraction, bt_fraction).astype(np.float32),
        "annular_uncertainty": np.maximum(np.nan_to_num(br_unc), np.nan_to_num(bt_unc)).astype(np.float32),
        "annular_valid_fraction_grid": _expand_profile(np.minimum(br_fraction, bt_fraction), radius, rb).astype(np.float32),
        "annular_uncertainty_grid": _expand_profile(np.maximum(np.nan_to_num(br_unc), np.nan_to_num(bt_unc)), radius, rb).astype(np.float32),
        "profile_r": br_profile.astype(np.float32),
        "profile_t": bt_profile.astype(np.float32),
        "sigma_pixels": float(sigma_px),
        "reconstruction_error": reconstruction_error,
    }


def components_to_uv(radial: np.ndarray, tangential: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return radial_tangential_to_uv(radial, tangential, x_km, y_km)
