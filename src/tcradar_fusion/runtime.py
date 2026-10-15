from __future__ import annotations
import gc,json,copy
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import MissionStore, PairDataset, PairRecord, pair_collate, pair_seed


class LazyRecordMissionStore(MissionStore):
    """Build stable pair indices and load mission fields only when requested."""

    def records(
        self,
        storms: set[str] | None,
        split: str,
        cfg: dict[str, Any],
        epoch: int = 0,
    ) -> list[PairRecord]:
        masks = cfg["mask"]
        count_key = "masks_per_pair_train" if split == "train" else "masks_per_pair_validation"
        if split in {"locked_test", "test"}:
            count_key = "masks_per_pair_test"
        n_masks = int(masks.get(count_key, 1))
        rows: list[PairRecord] = []
        for mission_id in self.mission_ids(storms):
            group = self.groups[mission_id]
            storm_key = str(self.merged.loc[mission_id, "storm_key"])
            n_swaths = len(group)
            fixed_rows = (
                self.fixed_masks[self.fixed_masks["mission_ID"].astype(str) == str(mission_id)]
                if split not in {"train", "archive", "archive_inference"} and not self.fixed_masks.empty
                else pd.DataFrame()
            )
            mask_iter = (
                [
                    (int(row["mask_index"]), int(row["seed"]), str(row["mask_type"]))
                    for _, row in fixed_rows.sort_values("mask_index").iterrows()
                ]
                if not fixed_rows.empty
                else None
            )
            for ref_index in range(n_swaths):
                if mask_iter is None:
                    current_masks = [
                        (
                            i,
                            pair_seed(int(cfg["project"]["seed"]), mission_id, ref_index, i),
                            str(masks["types"][i % len(masks["types"])]),
                        )
                        for i in range(n_masks)
                    ]
                else:
                    current_masks = mask_iter
                for mask_index, fixed_seed, fixed_type in current_masks:
                    if split in {"archive", "archive_inference"}:
                        rows.append(PairRecord(mission_id, storm_key, split, ref_index, 0, 0, "none"))
                    else:
                        seed = int(fixed_seed) if not fixed_rows.empty else pair_seed(
                            int(cfg["project"]["seed"]), mission_id, ref_index, mask_index, epoch if split == "train" else 0
                        )
                        rows.append(PairRecord(mission_id, storm_key, split, ref_index, mask_index, seed, fixed_type))
        return rows


def _storm_list(parent: dict[str, Any], name: str) -> list[str]:
    path = Path(parent["paths"]["output_root"]) / "splits" / f"{name}_storms.csv"
    return pd.read_csv(path)["storm_key"].astype(str).tolist()


def _normalization(parent: dict[str, Any]) -> dict[str, Any]:
    path = Path(parent["paths"]["output_root"]) / "preprocessing" / "normalization.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_fold(parent: dict[str, Any], smoke: bool = False) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    train_all = _storm_list(parent, "train")
    validation_all = _storm_list(parent, "validation")
    test_all = _storm_list(parent, "test")
    train = train_all[:1] if smoke else train_all
    validation = validation_all[:1] if smoke else validation_all
    test = [] if smoke else test_all
    fold = {
        "train_storms": sorted(train),
        "validation_storms": sorted(validation),
        "locked_test_storms": sorted(test),
        "development_storms": sorted(set(train + validation)),
    }
    full = {
        "train_storms": sorted(train_all),
        "validation_storms": sorted(validation_all),
        "locked_test_storms": sorted(test_all),
    }
    return fold, full


def _loader(
    store: MissionStore,
    records: list[PairRecord],
    cfg: dict[str, Any],
    normalization: dict[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        PairDataset(store, records, cfg, normalization),
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        collate_fn=pair_collate,
    )


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _checkpoint_state(
    temporal: torch.nn.Module,
    fusion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "temporal": _cpu_state_dict(temporal),
        "fusion": _cpu_state_dict(fusion),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "fold": 0,
        "epoch": int(epoch),
        "config": str(cfg["_config_path"]),
    }


def _release_store_memory(store: MissionStore) -> None:
    """Release cached mission arrays after a completed computation."""
    for name in ("_decompositions", "_missions"):
        cache = getattr(store, name, None)
        if isinstance(cache, dict):
            cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _archive_records(store: MissionStore, cfg: dict[str, Any]) -> list[PairRecord]:
                                                                             
                                                                            
                                                                            
                                                               
    rows: list[PairRecord] = []
    for mission_id in store.mission_ids(None):
        group = store.groups[mission_id]
        reference_rows = group.index[group["is_primary_reference"].astype(bool)]
        if len(reference_rows) != 1:
            raise ValueError(f"Mission {mission_id} must have one primary reference")
        reference_index = int(reference_rows[0])
        rows.append(
            PairRecord(
                mission_id=str(mission_id),
                storm_key=str(store.merged.loc[mission_id, "storm_key"]),
                split="archive_inference",
                pseudo_reference_index=reference_index,
                mask_index=0,
                mask_seed=0,
                mask_type="none",
            )
        )
    return rows

