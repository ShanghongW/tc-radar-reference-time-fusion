"""Quality control, single-split fitting, evaluation and product generation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ensure_dirs, load_config, set_reproducible
from .data_config import load_config as load_data_config
from .qc import build_fusion_qc, write_qc_tables
from .splits import assign_storm_splits, propagate_split_to_swaths, write_split_tables
from .masking import build_fixed_mask_manifest
from .cache import build_swath_cache
from .normalization import compute_training_normalization, write_normalization
from .models import TemporalNet
from .fusion import FusionNet
from .losses import temporal_loss, fusion_loss
from .pipeline import _move, _forward, _permutation_difference, evaluate_loader, _write_archive_product
from .runtime import LazyRecordMissionStore, _build_fold, _loader, _normalization, _checkpoint_state, _release_store_memory, _archive_records


def prepare(config_path: str, cache: bool = False) -> dict:
    cfg = load_data_config(config_path)
    merged, swaths = build_fusion_qc(cfg)
    merged, storms = assign_storm_splits(merged, cfg["split"])
    swaths = propagate_split_to_swaths(swaths, merged)
    root = Path(cfg["paths"]["output_root"])
    write_qc_tables(merged, swaths, root)
    write_split_tables(storms, merged, root)
    build_fixed_mask_manifest(merged, cfg, root)
    write_normalization(compute_training_normalization(swaths, cfg), root)
    if cache:
        build_swath_cache(swaths, cfg)
    passed = merged[merged["fusion_qc_pass"]]
    return {"missions": len(passed), "storms": passed["storm_key"].nunique(), "swaths": int(passed["n_swaths_good"].sum()), "output": str(root)}


def _context(config_path: str):
    cfg, parent = load_config(config_path)
    ensure_dirs(cfg)
    set_reproducible(int(cfg["project"]["seed"]), bool(cfg["runtime"].get("deterministic", True)))
    device = torch.device("cuda" if cfg["training"].get("use_cuda", True) and torch.cuda.is_available() else "cpu")
    fold, _ = _build_fold(parent)
    roles = [set(fold[k]) for k in ["train_storms", "validation_storms", "locked_test_storms"]]
    if any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Storms occur in more than one split")
    return cfg, parent, device, fold


def _models(cfg, device, checkpoint: str | None = None):
    temporal, fusion = TemporalNet(cfg).to(device), FusionNet(cfg).to(device)
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        temporal.load_state_dict(state["temporal"])
        fusion.load_state_dict(state["fusion"])
    return temporal, fusion, state


def train(config_path: str, epochs: int | None = None, resume: str | None = None) -> dict:
    cfg, parent, device, fold = _context(config_path)
    root = Path(cfg["output_root"])
    (root / "split_roles.json").write_text(json.dumps(fold, indent=2))
    store, norm = LazyRecordMissionStore(parent), _normalization(parent)
    train_records = store.records(set(fold["train_storms"]), "train", cfg)
    selection_records = store.records(set(fold["validation_storms"]), "validation", cfg)
    if not train_records or not selection_records:
        raise ValueError("Fitting and selection samples must both be nonempty")
    temporal, fusion, state = _models(cfg, device, resume)
    optimizer = torch.optim.AdamW(list(temporal.parameters()) + list(fusion.parameters()), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    start = 0
    best = float("inf")
    best_epoch = -1
    history = []
    if state:
        start = int(state["epoch"]) + 1
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        best = float(state.get("best_validation_rmse", float("inf")))
        best_epoch = int(state.get("best_epoch", -1))
        if (root / "training_history.csv").exists():
            history = pd.read_csv(root / "training_history.csv").to_dict("records")
    end = int(epochs or cfg["training"]["epochs"])
    if end <= start:
        raise ValueError("The target epoch count must exceed the resumed epoch count")
    for epoch in range(start, end):
        loader = _loader(store, train_records, cfg, norm, shuffle=True, seed=int(cfg["project"]["seed"]) + 100000 + epoch)
        loader.dataset.set_epoch(epoch)
        temporal.train(); fusion.train()
        losses = []; begun = time.time()
        for batch_cpu in loader:
            batch = _move(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            temp, fused, product = _forward(batch, temporal, fusion, cfg)
            loss_t, _ = temporal_loss(batch, temp, cfg)
            permutation = _permutation_difference(batch, temp, fusion, cfg, product)
            loss_f, _ = fusion_loss(batch, temp, fused, product, cfg, permutation)
            loss = loss_t + loss_f
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fitting loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(temporal.parameters()) + list(fusion.parameters()), float(cfg["training"]["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            del batch_cpu, batch, temp, fused, product, loss_t, loss_f, loss, permutation
            _release_store_memory(store)
        _release_store_memory(store)
        selection_loader = _loader(store, selection_records, cfg, norm, shuffle=False, seed=int(cfg["project"]["seed"]))
        summary, rows = evaluate_loader(temporal, fusion, selection_loader, cfg, device, "validation", 0)
        score = float(summary.get("reference_time_rmse", float("inf")))
        if not np.isfinite(score):
            raise FloatingPointError("No finite model-selection score")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.time() - begun, **summary}
        history.append(row)
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        snapshot = _checkpoint_state(temporal, fusion, optimizer, epoch, cfg)
        snapshot.update(best_validation_rmse=best, best_epoch=best_epoch)
        torch.save(snapshot, root / "checkpoints/last.pt")
        if improved:
            torch.save(snapshot, root / "checkpoints/best.pt")
        pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
        pd.DataFrame(rows).to_csv(root / "selection_metrics.csv", index=False)
        _release_store_memory(store)
        print(json.dumps(row), flush=True)
    return {"best_epoch": best_epoch, "best_validation_rmse": best, "checkpoint": str(root / "checkpoints/best.pt")}


def evaluate(config_path: str, checkpoint: str, split: str = "test") -> dict:
    cfg, parent, device, fold = _context(config_path)
    keys = {"train": "train_storms", "validation": "validation_storms", "test": "locked_test_storms"}
    temporal, fusion, _ = _models(cfg, device, checkpoint)
    store = LazyRecordMissionStore(parent)
    records = store.records(set(fold[keys[split]]), "locked_test" if split == "test" else split, cfg)
    loader = _loader(store, records, cfg, _normalization(parent), shuffle=False, seed=int(cfg["project"]["seed"]))
    summary, rows = evaluate_loader(temporal, fusion, loader, cfg, device, split, 0)
    frame = pd.DataFrame(rows)
    out = Path(cfg["output_root"]) / "evaluation"; out.mkdir(exist_ok=True)
    frame.to_csv(out / f"{split}_samples.csv", index=False)
    product_rows = frame[frame["method"].isin(["raw_equal", "harmonized_equal", "reference_time"])]
    mission = product_rows.groupby(["storm_key", "mission_id", "method"], as_index=False)[["rmse", "mae", "speed_rmse"]].mean()
    storm = mission.groupby(["storm_key", "method"], as_index=False)[["rmse", "mae", "speed_rmse"]].mean()
    mission.to_csv(out / f"{split}_missions.csv", index=False)
    storm.to_csv(out / f"{split}_storms.csv", index=False)
    storm.groupby("method")[["rmse", "mae", "speed_rmse"]].mean().to_csv(out / f"{split}_storm_equal.csv")
    return {"split": split, "samples": len(frame), "storms": int(storm.storm_key.nunique()), "output": str(out)}


def infer(config_path: str, checkpoint: str, mission_id: str | None = None) -> dict:
    cfg, parent, device, _ = _context(config_path)
    temporal, fusion, _ = _models(cfg, device, checkpoint)
    temporal.eval(); fusion.eval()
    store = LazyRecordMissionStore(parent)
    records = _archive_records(store, cfg)
    if mission_id is not None:
        records = [r for r in records if r.mission_id == mission_id]
        if not records:
            raise KeyError(f"Mission not in the retained catalog: {mission_id}")
    rows = []
    for record in records:
        batch_cpu = next(iter(_loader(store, [record], cfg, _normalization(parent), shuffle=False, seed=int(cfg["project"]["seed"]))))
        batch = _move(batch_cpu, device)
        with torch.no_grad():
            temp, _, product = _forward(batch, temporal, fusion, cfg)
        path = _write_archive_product(cfg, Path(cfg["output_root"]), batch_cpu, temp, product, Path(checkpoint))
        rows.append({"mission_id": record.mission_id, "product": str(path)})
        del batch_cpu, batch, temp, product
        _release_store_memory(store)
    pd.DataFrame(rows).to_csv(Path(cfg["output_root"]) / "products_manifest.csv", index=False)
    return {"products": len(rows), "directory": str(Path(cfg["output_root"]) / "products")}
