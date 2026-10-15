from __future__ import annotations
import numpy as np


def polar_geometry(x_km: np.ndarray, y_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.asarray(x_km, float), np.asarray(y_km, float), indexing="xy")
    return np.hypot(xx, yy), np.arctan2(yy, xx)


def uv_to_radial_tangential(
    u: np.ndarray, v: np.ndarray, x_km: np.ndarray, y_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, theta = polar_geometry(x_km, y_km)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radial = u * cos_t + v * sin_t
    tangential = -u * sin_t + v * cos_t
    return radial.astype(np.float32), tangential.astype(np.float32)


def radial_tangential_to_uv(
    radial: np.ndarray, tangential: np.ndarray, x_km: np.ndarray, y_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, theta = polar_geometry(x_km, y_km)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = radial * cos_t - tangential * sin_t
    v = radial * sin_t + tangential * cos_t
    return u.astype(np.float32), v.astype(np.float32)


def derivatives(
    u: np.ndarray, v: np.ndarray, x_km: np.ndarray, y_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(x_km))) * 1000.0
    dy = float(np.nanmedian(np.diff(y_km))) * 1000.0
    filled_u = np.where(np.isfinite(u), u, 0.0)
    filled_v = np.where(np.isfinite(v), v, 0.0)
    dudx = np.gradient(filled_u, dx, axis=-1)
    dudy = np.gradient(filled_u, dy, axis=-2)
    dvdx = np.gradient(filled_v, dx, axis=-1)
    dvdy = np.gradient(filled_v, dy, axis=-2)
    return (dvdx - dudy).astype(np.float32), (dudx + dvdy).astype(np.float32)

