# TC-RADAR reference-time fusion

This repository implements a reference-time wind analysis from sequential airborne Doppler radar swaths in tropical cyclones. A prescribed observed swath defines the reference time. Other swaths receive signed-time wind adjustments at their original grid points and are combined through learned error covariance and reliability. The primary-reference observations are retained, and locations without radar support remain missing.

## Included algorithms

- Mission and swath quality control, duplicate handling and coverage-weighted reference selection.
- Storm-level fitting, model-selection and held-out groups.
- Source-only annular background, smoothed asymmetry and residual decomposition.
- Joint fitting of a pointwise temporal-tendency network and an order-invariant combination model.
- Coverage-aware reference masking and fixed evaluation masks.
- Full 2-by-2 precision fusion, reference preservation and NetCDF product export.
- Fixed-checkpoint reconstruction metrics with mission-then-storm aggregation.
- STAR SAR geographic colocation and common-support structural comparison.

The repository contains source code, parameter templates and an in-memory computational example. Observations, quality-control catalogs, fitted model weights and generated products are supplied or generated locally.

## Install

Python 3.10 or newer is required. Install an appropriate PyTorch build for the available CPU or GPU, then install the package:

```bash
python -m pip install -e .
```

For SAR comparison and its figures:

```bash
python -m pip install -e '.[sar]'
```

Run the data-free example:

```bash
python examples/synthetic_inference.py
```

It uses random model weights and generated arrays only; it demonstrates execution and array organization, not reconstruction skill.

## Prepare your data

Obtain TC-RADAR v3m observations and the associated mission metadata separately. Edit the paths in `configs/data.json`. Paths are resolved relative to the configuration file. The variable names, coordinate contract and catalog fields are described in [docs/data_contract.md](docs/data_contract.md).

```bash
tcradar-fusion show-config --config configs/model.json
tcradar-fusion prepare --data-config configs/data.json
```

Add `--cache` to prepare compressed local swath caches. The preparation command writes the quality-control tables, storm assignments, fixed mask bank and fitting-only normalization under `outputs/prepared/`.

## Fit and select a model

```bash
tcradar-fusion train --config configs/model.json --epochs 16
```

The fitting group supplies gradients. The validation group selects `outputs/model/checkpoints/best.pt`. The test group is used only by an explicit evaluation command. To continue fitting, increase the total epoch count and resume from the local last checkpoint:

```bash
tcradar-fusion train --config configs/model.json --epochs 18 --resume outputs/model/checkpoints/last.pt
```

Fitting processes one mission/pseudo-reference sample per batch because the number of sources varies. The implementation releases cached fields between completed computations to bound memory use.

## Evaluate and construct products

```bash
tcradar-fusion evaluate --config configs/model.json --checkpoint outputs/model/checkpoints/best.pt --split test
tcradar-fusion infer --config configs/model.json --checkpoint outputs/model/checkpoints/best.pt
```

Add `--mission YOUR_MISSION_ID` to infer one mission. Evaluation writes sample, mission and storm tables; the final summary weights storms equally. Inference uses the catalog's primary reference and writes one NetCDF file per mission.

## Optional SAR comparison

```bash
tcradar-fusion compare-sar --data-config configs/data.json --mission YOUR_MISSION_ID --sar-file /path/to/STAR_SAR_product.h5 --product-dir outputs/model/products
```

The comparison adds the motion translation to storm-relative radar winds, samples the geographic common support and forms area-weighted super-observations. The default horizontal aggregation is 3 km; radar height follows the lowest configured analysis level, normally 0.5 km. SAR near-surface and airborne winds are compared structurally across observing heights.

## Parameters and outputs

`configs/model.json` contains network widths, tendency/modulation bounds, regularization scales, losses, learning rate and seeds. See [docs/method.md](docs/method.md) for the operator and output interpretation. Existing checkpoints may be loaded only with their matching architecture and parameter configuration.

All local datasets, checkpoints, caches, logs, evaluation tables and NetCDF products are excluded by `.gitignore`.
