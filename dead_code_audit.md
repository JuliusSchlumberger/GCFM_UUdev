# Dead Code / Unused Function Audit

Scope: `config/` and `workflow/` only. Generated 2026-08-05.
Method: cross-referenced every function/class/import/script/rule/config key against usage across the whole `workflow/` and `config/` tree. Key items were spot-checked directly (grep) rather than taken on trust.

## 1. Dead functions (safe to delete, high confidence)

No references found anywhere outside their own `def` line.

| File | Line | Function |
|---|---|---|
| `workflow/src/raster.py` | 253 | `clip_raster` |
| `workflow/src/raster.py` | 19 | `resample_to_utm_array` |
| `workflow/src/raster.py` | 110 | `load_raster_to_utm_array` |
| `workflow/src/river_network.py` | 925 | `sample_dem_near_river` |
| `workflow/src/river_depth_calibration.py` | 120 | `place_reach_midpoint_observations` |
| `workflow/src/river_depth_calibration.py` | 323 | `sample_model_bed_at_points` |
| `workflow/src/extreme_values.py` | 744 | `analyse_cell_gev_only` (sibling `analyse_cell` is the one actually used) |
| `workflow/src/plots.py` | 582 | `plot_global_protection_map` |

Spot-checked `clip_raster` / `resample_to_utm_array` / `load_raster_to_utm_array` directly — confirmed no call sites anywhere in `workflow/`.

**Medium-high confidence** (implemented and documented, but no call site found):

| File | Line | Function | Note |
|---|---|---|---|
| `workflow/src/extreme_values.py` | 1420 | `plot_grdc_overview` | Diagnostic plot; script only calls `plot_bias_correction`/`plot_cell_diagnostics` from this module. |

## 2. Unused imports

| File | Line | Import |
|---|---|---|
| `workflow/scripts/14_run_spinup.py` | 45 | `import os` |
| `workflow/scripts/17_flood_metrics.py` | 7 | `import geopandas as gpd` |

Verified: no `os.` or `gpd.` usage anywhere else in either file.

## 3. Orphaned bytecode cache

| File | Note |
|---|---|
| `workflow/src/__pycache__/quadtree_refinement.cpython-310.pyc` | No matching `quadtree_refinement.py` exists (removed per commit `2e14535`, "remove quadtree"). Safe to delete — `__pycache__` is regenerated automatically anyway. |
| `workflow/src/__pycache__/quadtree_refinement.cpython-311.pyc` | Same. |

## 4. Archive folder — inconsistent, worth a decision

`workflow/archive/datum_correction/` (3 files) is correctly unreferenced/inert — nothing imports it. But its header comment claims the EGM2008→GOCO06s/MDT datum-correction logic was *removed* from the elevation pipeline. In reality, the **live** code still has and actively uses equivalent/successor functions:
- `workflow/src/raster.py:470` `compute_geoid_offset_arr`
- `workflow/src/plots.py:2645/2700/2757` `plot_geoid_offset`, `plot_mdt_ocean`, `plot_datum_correction_delta`

...all called from `workflow/scripts/05a_get_elevation.py`. So either the archive is stale documentation (functionality was reinstated after archiving, notes never updated), or the archive should just be deleted since it's now fully superseded/duplicated by live code. Recommend deleting the archive folder rather than keeping misleading comments around.

## 5. Checked and found clean (no action needed)

- **Scripts**: all 23 files in `workflow/scripts/*.py` are referenced by a `script:` directive in some `workflow/rules/*.smk` file. None orphaned.
- **Rule files**: all 20 `workflow/rules/*.smk` files are `include:`-d in `workflow/Snakefile`. None orphaned.
- **Duplicated logic**: no function/class defined twice across `workflow/src/*.py`.
- **`config/config.yml`, `config/scenarios.yml`, `config/data_catalogue.yml`**: top-level sections are all read somewhere in `workflow/`. (Note: individual dataset entries inside `data_catalogue.yml`'s `datasets:` list were not exhaustively checked one-by-one against every catalogue lookup call — flagging as not fully covered, but no evidence of dead entries either.)
- Underscore-prefixed helpers in `extreme_values.py`, `plots.py`, `postprocessing.py`, `protection_levels.py`, `protection_weir.py`, `river_burn.py`, `river_forcing.py`, `river_network.py`, `surge.py` are all called from within their own module — normal private-helper pattern, not dead.

## Suggested next step

1. Delete the 8 dead functions in §1 (or the 9 if you agree on `plot_grdc_overview`).
2. Remove the 2 unused imports in §2.
3. Delete the 2 orphaned `.pyc` files in §3 (cosmetic — `__pycache__` regenerates).
4. Delete `workflow/archive/datum_correction/` (§5) or rewrite its stale comment — currently misleading about what's actually live.
