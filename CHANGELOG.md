# Changelog

## 2026-10-15: Temporal-scale configuration

- Make the supplied `T_B = T_L = T_S = 12 h` configuration explicit in the method description, README and temporal-operator component ordering.
- The numerical settings remain `[12.0, 12.0, 12.0]`; no model weights or observational results are changed.

## 0.1.0

- Standalone package for preparation, joint single-split fitting, masked reconstruction, reference-time inference and SAR comparison.
- Relative input and output paths are resolved from configuration files.
- Source permutation consistency applies the same permutation to candidate fields, time offsets, source coverage and context features. This concerns the auxiliary fitting regularizer; the temporal and fusion forward operators are unchanged.
- Observation files, fitted weights and generated diagnostics are maintained outside the repository.
