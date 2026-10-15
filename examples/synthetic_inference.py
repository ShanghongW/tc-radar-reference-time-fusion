"""Run the model on in-memory winds; no observational data or files are used."""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from tcradar_fusion.data import PairRecord, build_pair, pair_collate
from tcradar_fusion.multiscale import decompose_source
from tcradar_fusion.pipeline import _forward
from tcradar_fusion.models import TemporalNet
from tcradar_fusion.fusion import FusionNet


def example_batch(cfg):
    x = np.linspace(-20, 20, 21, dtype=np.float32)
    y = x.copy(); z = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    u0 = np.broadcast_to(-0.5 * yy, (3, 21, 21)).copy()
    v0 = np.broadcast_to(0.5 * xx, (3, 21, 21)).copy()
    u = np.stack([u0 - 0.3, u0, u0 + 0.3])
    v = np.stack([v0 + 0.2, v0, v0 - 0.2])
    u[0, :, :, :4] = np.nan; v[0, :, :, :4] = np.nan
    u[2, :, :, -4:] = np.nan; v[2, :, :, -4:] = np.nan
    mission = SimpleNamespace(u=u, v=v, x=x, y=y, z=z, times_hours=np.array([-1, 0, 1], dtype=np.float32), swath_ids=["source_earlier", "reference", "source_later"], coverages=np.array([0.8, 1.0, 0.8], dtype=np.float32))
    class Store:
        def mission(self, mission_id):
            return mission
        def component(self, mission_id, source_index, settings):
            mask = np.isfinite(u[source_index]) & np.isfinite(v[source_index])
            return decompose_source(u[source_index], v[source_index], mask, x, y, settings["data"])
    record = PairRecord("synthetic", "synthetic_storm", "validation", 1, 0, 7, "block")
    pair = build_pair(record, Store(), cfg, {"u_std": 10, "v_std": 10})
    return pair_collate([pair])


def main():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/model.json").read_text())
    torch.manual_seed(7)
    batch = example_batch(cfg)
    temporal, fusion = TemporalNet(cfg), FusionNet(cfg)
    temporal.eval(); fusion.eval()
    with torch.no_grad():
        aligned, _, product = _forward(batch, temporal, fusion, cfg)
    print(json.dumps({"example": "in-memory synthetic winds with randomly initialized weights", "input_shape": list(batch["features"].shape), "candidate_support_unchanged": bool(torch.equal(aligned["source_mask"], batch["source_support"])), "hidden_cells": int(batch["hidden_mask"].sum()), "product_shape": list(product["fused_r"].shape)}))


if __name__ == "__main__":
    main()
