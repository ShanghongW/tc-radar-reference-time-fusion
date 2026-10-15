from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="TC-RADAR reference-time wind analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Build quality-control catalogs, storm split, fixed masks and fitting-only normalization")
    p.add_argument("--data-config", required=True)
    p.add_argument("--cache", action="store_true", help="Cache retained wind fields in the configured output directory")
    p = commands.add_parser("train", help="Jointly fit the temporal and combination models")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", help="Local last.pt checkpoint")
    p = commands.add_parser("evaluate", help="Reconstruct withheld reference observations with a fixed checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p = commands.add_parser("infer", help="Write one product per retained mission using its primary reference")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mission", help="Optional mission identifier")
    p = commands.add_parser("compare-sar", help="Compare a fixed product with a STAR SAR snapshot")
    p.add_argument("--data-config", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--sar-file", required=True)
    p.add_argument("--product-dir")
    p.add_argument("--resolution-km", type=float, default=3.0)
    p = commands.add_parser("show-config", help="Validate and display resolved configuration paths")
    p.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "show-config":
        from .config import load_config
        cfg, parent = load_config(args.config)
        result = {"model_output": cfg["output_root"], "data_paths": parent["paths"], "time_scales_hours": cfg["temporal_model"]["time_scales_hours"]}
    elif args.command == "compare-sar":
        from .satellite import run_satellite_validation
        result = run_satellite_validation(args.sar_file, args.data_config, mission_id=args.mission, resolution_km=args.resolution_km, fused_product_dir=args.product_dir)
    else:
        from . import workflow
        if args.command == "prepare":
            result = workflow.prepare(args.data_config, args.cache)
        elif args.command == "train":
            result = workflow.train(args.config, args.epochs, args.resume)
        elif args.command == "evaluate":
            result = workflow.evaluate(args.config, args.checkpoint, args.split)
        else:
            result = workflow.infer(args.config, args.checkpoint, args.mission)
    print(json.dumps(result, indent=2, default=str))
