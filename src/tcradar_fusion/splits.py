from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


def _target_sizes(n: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: n * fractions[name] for name in SPLITS}
    sizes = {name: int(math.floor(raw[name])) for name in SPLITS}
    remainder = n - sum(sizes.values())
    order = sorted(SPLITS, key=lambda name: (raw[name] - sizes[name], fractions[name]), reverse=True)
    for name in order[:remainder]:
        sizes[name] += 1
    if n >= 3:
        for name in ("validation", "test"):
            if sizes[name] == 0:
                donor = max(SPLITS, key=lambda key: sizes[key])
                sizes[donor] -= 1
                sizes[name] += 1
    return sizes


def _storm_table(merged_pass: pd.DataFrame) -> pd.DataFrame:
    df = merged_pass.copy()
    df["year"] = pd.to_datetime(df["mission_time"], errors="coerce").dt.year
    df["coverage_proxy"] = pd.to_numeric(df["core_minimum_quadrant_coverage_2km"], errors="coerce")
    rows = []
    for key, group in df.groupby("storm_key", sort=True):
        classes = group["fischer_class_12h"].value_counts()
        rows.append({
            "storm_key": key,
            "storm_name": ";".join(sorted(set(group["storm_name"].dropna().astype(str)))),
            "n_missions": len(group),
            "n_swaths_good": group["n_swaths_good"].sum(),
            "n_RI": int(classes.get("RI", 0)),
            "n_SI": int(classes.get("SI", 0)),
            "n_NI": int(classes.get("NI", 0)),
            "vmax_mean_kt": group["vmax_0h_kt"].mean(),
            "year_mean": group["year"].mean(),
            "rmw_mean_km": group["rmw_2km_km"].mean(),
            "coverage_mean": group["coverage_proxy"].mean(),
            "span_mean_hours": group["good_swath_time_span_hours"].mean(),
        })
    return pd.DataFrame(rows).sort_values("storm_key", kind="mergesort").reset_index(drop=True)


def assign_storm_splits(merged: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    passed = merged[merged["fusion_qc_pass"]].copy()
    storms = _storm_table(passed)
    if storms.empty:
        raise RuntimeError("No Fusion-QC-passing storms are available for splitting")
    fractions = {
        "train": float(cfg["train_fraction"]),
        "validation": float(cfg["validation_fraction"]),
        "test": float(cfg["test_fraction"]),
    }
    total_fraction = sum(fractions.values())
    fractions = {key: value / total_fraction for key, value in fractions.items()}
    sizes = _target_sizes(len(storms), fractions)
    rng = np.random.default_rng(int(cfg["seed"]))
    metrics = ["n_missions", "n_swaths_good", "n_RI", "n_SI", "n_NI", "vmax_mean_kt", "year_mean", "rmw_mean_km", "coverage_mean", "span_mean_hours"]
    values = storms[metrics].fillna(storms[metrics].median(numeric_only=True)).fillna(0.0).to_numpy(float)
    targets = np.stack([values.sum(axis=0) * fractions[name] for name in SPLITS])
    scale = np.nanstd(values, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    feature_weights = np.asarray([2.0, 1.5, 9.0, 5.0, 5.0, 0.5, 0.4, 0.7, 0.7, 0.7])

    def assignment_score(labels: np.ndarray) -> float:
        aggregate = np.stack([values[labels == index].sum(axis=0) for index in range(len(SPLITS))])
        denominator = np.abs(targets) + 0.25 * scale[None, :] + 1.0
        score = float(np.mean(feature_weights[None, :] * np.square((aggregate - targets) / denominator)))
        for metric_index in (2, 3, 4):
            if values[:, metric_index].sum() > 0:
                score += 4.0 * float(np.sum(aggregate[:, metric_index] == 0))
        return score

    best_labels: np.ndarray | None = None
    best_score = float("inf")
    boundaries = np.cumsum([0] + [sizes[name] for name in SPLITS])
    for _ in range(int(cfg.get("search_iterations", 30000))):
        permutation = rng.permutation(len(storms))
        labels = np.empty(len(storms), dtype=np.int8)
        for split_index in range(len(SPLITS)):
            labels[permutation[boundaries[split_index] : boundaries[split_index + 1]]] = split_index
        score = assignment_score(labels)
        if score < best_score:
            best_score = score
            best_labels = labels.copy()
    assert best_labels is not None
    assigned: dict[str, list[int]] = {
        name: np.flatnonzero(best_labels == index).astype(int).tolist()
        for index, name in enumerate(SPLITS)
    }
    split_map: dict[str, str] = {}
    for name, indices in assigned.items():
        for idx in indices:
            split_map[str(storms.iloc[idx]["storm_key"])] = name
    storms["split"] = storms["storm_key"].map(split_map)
    out = merged.copy()
    out.loc[out["fusion_qc_pass"], "split"] = out.loc[out["fusion_qc_pass"], "storm_key"].map(split_map)
    assert not out[out["fusion_qc_pass"]].groupby("storm_key")["split"].nunique().gt(1).any()
    return out, storms


def propagate_split_to_swaths(swath: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    split_map = merged.set_index("mission_ID")["split"]
    out = swath.copy()
    out["split"] = out["mission_ID"].map(split_map).fillna("excluded")
    return out


def write_split_tables(storms: pd.DataFrame, merged: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    out = Path(output_root) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = out / f"{split}_storms.csv"
        storms[storms["split"] == split].sort_values("storm_key").to_csv(path, index=False, encoding="utf-8")
        paths[split] = path
    manifest = out / "split_manifest.csv"
    storms.sort_values(["split", "storm_key"], kind="mergesort").to_csv(manifest, index=False, encoding="utf-8")
    paths["manifest"] = manifest
    summary_rows = []
    passed = merged[merged["fusion_qc_pass"]]
    for split in SPLITS:
        part = passed[passed["split"] == split]
        counts = part["fischer_class_12h"].value_counts()
        summary_rows.append({
            "split": split,
            "n_storms": part["storm_key"].nunique(),
            "n_missions": len(part),
            "n_good_swaths": int(part["n_swaths_good"].sum()),
            "n_RI": int(counts.get("RI", 0)),
            "n_SI": int(counts.get("SI", 0)),
            "n_NI": int(counts.get("NI", 0)),
            "vmax_mean_kt": part["vmax_0h_kt"].mean(),
            "rmw_mean_km": part["rmw_2km_km"].mean(),
            "coverage_mean": part["core_minimum_quadrant_coverage_2km"].mean(),
            "span_mean_hours": part["good_swath_time_span_hours"].mean(),
        })
    summary = out / "split_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary, index=False, encoding="utf-8")
    paths["summary"] = summary
    return paths
