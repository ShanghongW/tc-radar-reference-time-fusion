# Data and coordinate contract

Inputs are TC-RADAR recentered eastward and northward winds with coordinate variables `eastward_distance`, `northward_distance` and `height`, measured in kilometers. The source NetCDF arrays use the archive case and horizontal/vertical dimensions. The loader normalizes each swath to `(z, y, x)` and organizes a mission as `(swath, z, y, x)`.

The product domain is defined in `configs/data.json`: radius at most 150 km, heights 0.5–18 km, and a fixed 2-km storm center. Sources in one mission must have the same recentered grid. Paired finite u/v observations define support; zero wind is observed. No field or source mask is spatially displaced.

Preparation requires the original merged and swath files and `qc_all.csv`. The archived SHIPS fields are read for sample selection and characterization. Future intensity-change categories are not temporal-network predictors or reconstruction targets.

Preparation generates these files locally:

| Directory | Content |
|---|---|
| `qc/` | Complete merged/swath screening tables and retained subsets |
| `splits/` | Storm assignments for train, validation and test, plus summaries |
| `masks/` | Fixed evaluation-mask descriptors |
| `preprocessing/` | Normalization determined only from fitting swaths |
| `cache/` | Optional local swath arrays and a cache manifest |

The mission catalog contains `mission_ID`, `storm_key`, `fusion_qc_pass`, `split`, `reference_swath_ID` and the selection metadata. The swath catalog contains `swath_ID`, `mission_ID`, `source_path`, `source_file`, `swath_case_index`, `swath_midpoint_time`, `swath_coverage_2km`, `swath_qc_pass`, `parent_mission_fusion_qc_pass`, and `is_primary_reference`. These catalogs are generated from local observations rather than distributed with the code.

The model tensor shape is `(batch, source, channel, z, y, x)`. Only visible reference observations contribute reference-derived inputs. Hidden-reference targets are restricted to originally observed reference cells with at least one unchanged source observation.

The signed time separation is `reference midpoint minus source midpoint`, in hours. Product timestamps and source offsets refer to swath midpoints. No within-swath ray-time adjustment is performed.
