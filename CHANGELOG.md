# Changelog

Newest changes first. See `Reference_memory.txt` for the current, up-to-date
description of how the pipeline works; this file only describes *what changed
and why*.

# 2028-08-21 Added adaptation effectiveness assessment (KL)

New pipeline stage for evaluating coastal/river flood adaptation strategies against each scenario, on top of the existing baseline SFINCS workflow. Two config files define the search space: `config/measures.yml` is the menu -- every measure type (`offshore_barrier`, `pumps`, `nbs_land_reclamation`, river levees, retreat, ...) grouped by main strategy (Advance / Grey protect-open / NBS protect-open / Retreat), each with parameter bounds and whether it's supported in `preprocessing` and/or `postprocessing` mode; `config/adaptation_strategies.yml` is the composition -- named strategies, each a set of measures with concrete parameter values, independent of which method runs them (method is chosen at the CLI, validated against `measures.yml`).

**Added**: two ways to apply a strategy. The pre-processing method (`18a_adapt_pre.smk`/`.py`, `src/adaptation_method_pre.py`) mutates a copy of the basin's SFINCS skeleton in place per measure -- weirs, drainage structures, subgrid rebuild, `retreat` land-use edits -- via `dispatch_rules()`, then reuses `13_build_sfincs.py`/`16_run_event.py`/`17_flood_metrics.py` unmodified against the adapted skeleton for a full physical re-simulation (never re-runs spin-up; ported from the author's separate `delta_model` project onto this repo's own `hydromt_sfincs` API). The post-processing method (`18b_adapt_post.smk`/`.py`, `src/adaptation_method_post.py`) applies measures directly to the existing baseline flood-depth raster -- e.g. `apply_offshore_barrier` compares barrier elevation against the scenario's max coastal water level and strips coastal/compound-attributed flooding pixels if the barrier holds -- no re-run needed.

The post-processing method needs to know WHERE flooding came from, so a new flood-source attribution mask (`18c_attribution_mask.smk`/`.py`, `src/attribution_plot.py`) classifies every flooded pixel as river-only / coastal-only / compound / baseline, by comparing each scenario's river-only and coastal-only counterpart runs against spin-up. A new rollup rule (`19_adapt_compare.smk`/`.py`) aggregates baseline/pre/post `flood_metrics.csv` across every scenario x strategy for a basin into one long-format `metrics_comparison.csv`.

Supporting changes: `src/protection_weir.py` gained `merge_weir_preserve_unmatched()` for combining adaptation-added weirs with existing ones; `rules/00_common.smk` extended with strategy/measure lookup helpers (`STRATEGY_DEFS`, `MEASURES_DEFS`, `strategy_measure_input_paths`, `attribution_counterparts`) and target-rule wiring for the new outputs; `tools/build_deltadtm_mask_vrt.py` added as a standalone tool to build a VRT mask from DeltaDTM; `tests/KL_analysis_waterlevel_animation.py` and `tests/KL_figures.py` added for reviewing water-level animations and figures.

`nbs_land_reclamation` and `water_retention` preprocessing measures are deliberately NOT ported yet -- `water_retention`'s `compute_excess_volume()` needs the attribution mask, which didn't exist until this PR; left as follow-up, noted directly in `adaptation_method_pre.py`'s own docstring.


# 2026-08-07e: fixed landuse_on_grid.tif/roughness_on_grid.tif row-order mismatch -- SFINCS needs y-ascending, sfincs_grid.json is stored y-descending (- JS)

User reported `zsini_on_grid` looked "rotated compared to other raster files... very small" after a real pipeline run, while confirming `landuse_on_grid.tif` itself "is oriented correctly and in the right dimensions" compared to other domain rasters. Root cause, found by directly instantiating `SfincsModel.grid.create_from_region()` and comparing its own transform against `sfincs_grid.json`'s stored one for basin 2433835: `hydromt.model.processes.create_grid_from_region` (the standalone function) returns a grid with y DESCENDING (row 0 = north, standard GDAL convention, `dy=-69.97`) -- but `SfincsModel.grid.create_from_region()` (used by every actual model-building script: rule modelled_depth_estimation's calibration model, build_sfincs_skeleton, build_sfincs) explicitly flips it (`if ds.raster.res[1] < 0: ds = ds.raster.flipud()`, hydromt_sfincs's own `regulargrid.py`) to y ASCENDING (row 0 = south, `dy=+69.97`) -- required by SFINCS's own m/n indexing convention (row index increases northward). Rule `build_sfincs_grid` (08c) deliberately calls the standalone function directly, without the flip, specifically to avoid instantiating a SfincsModel (which would leave behind an empty scratch directory) -- so `sfincs_grid.json` is stored in the OPPOSITE row-order convention from every real SfincsModel's own `sf.grid.data`.

09b_grid_align_landuse.py (2026-08-07b) read `sfincs_grid.json`'s raw transform directly to build `landuse_on_grid.tif`/`roughness_on_grid.tif` -- correct and self-consistent as a standalone file (matches other `sfincs_grid.json`-derived rasters like `elevation_conditioned_sfincs_grid.tif`, which is why the user found it "oriented correctly"), but every downstream consumer (rule 10's weir tracing/zsini, rule 13's roughness/weir diagnostics) reads it directly and assigns an actual SfincsModel's `sf.grid.data["dep"]` coords onto it with ZERO reprojection, by design (the whole point of the 09b refactor). That assignment silently mismatched row order: same pixel grid, same shape, opposite row 0. Before the 09b refactor, this was masked by `GridArrays.sample_landuse()`'s real `rasterio.warp.reproject` call, which is transform-aware and handles any orientation difference correctly regardless of sign convention -- replacing it with a raw file read broke that safety net.

**Fixed**: `09b_grid_align_landuse.py` now builds its own grid via `create_grid_from_region` + the SAME `flipud` check `SfincsModel.grid.create_from_region()` applies internally (replicated directly, not by instantiating a SfincsModel -- avoids reintroducing rule 08c's own scratch-directory problem), instead of reading `sfincs_grid.json`'s raw transform. Input swapped from `sfincs_grid` to `grid_resolution.json` + `domain_gpkg` (the same two inputs `create_grid_from_region` needs). No changes needed to any downstream consumer -- they were already written assuming SfincsModel-native orientation, which is now actually true.

Verified directly: built both grids for basin 2433835 and confirmed `new_transform == model_transform` and `new_shape == (531, 497)` matching `SfincsModel.grid.create_from_region()`'s own output exactly (`dy=+69.97`, `y0=4487731.97`, i.e. south-first). `tests/check_code_health.py` clean (61 files); full `snakemake -n` dry run through `compute_flood_metrics` resolves cleanly. Not yet re-verified against an actual rebuilt `zsini.tif`/weir overlay -- left to the user to confirm visually after the next pipeline run.

# 2026-08-07d: fixed a real runtime crash in 10_depth_estimation_modelled.py -- landuse_on_grid.tif vs elevation_conditioned_sfincs_grid.tif shape mismatch (- JS)

User's actual pipeline run for basin 2433835 crashed inside `modelled_depth_estimation`'s own calibration-model zsini construction: `ValueError: operands could not be broadcast together with shapes (531,497) (590,555)`. Root cause: the 2026-08-07b refactor moved this rule's OWN zsini construction (for its disposable calibration model, not production) to compare `landuse_on_grid.tif` (exact `sfincs_grid.json` shape, 531x497) directly against `elevation_conditioned_sfincs_grid.tif` (590x555) for the connected-component dry-out check -- these are NOT the same shape by design: rule `enforce_river_monotonicity` (09) deliberately EXTENDS the latter beyond the exact SFINCS grid bounds (snapped to the same resolution/phase) so HydroMT's `elevation.create()` has a safe margin for edge reprojection -- it was never meant to be pixel-identical in shape to the exact grid, only aligned in transform/phase. The earlier "confirmed pixel-identical" claims from the original zsini investigation were about a DIFFERENT comparison (`zsini_sea_cells_on_grid.tif` vs `sf.grid.data["dep"]`, both built via `grid.transform`/`grid.shape`, i.e. the model's own exact cropped grid) -- conflating the two was the mistake.

**Fixed**: moved this rule's own zsini construction from before grid creation to right after `sf.grid.create_from_region()` + mask setup, using `sf.grid.data["dep"]` (guaranteed the same exact shape as `landuse_on_grid.tif` -- both auto-UTM-fit from the same `domain_gpkg`/`grid_resolution.json`) for the connected-component check instead of `elevation_conditioned_sfincs_grid.tif`. Builds the in-memory `xr.DataArray` and calls `sf.initial_conditions.create(ini=<DataArray>, reproj_method="nearest")` directly -- same zero-reprojection pattern already used in `13_build_sfincs_skeleton.py` -- eliminating the `local_zsini` catalog-file indirection entirely (no more writing a scratch `zsini.tif` file just to register it in a data catalog before the model exists).

Verified: `tests/check_code_health.py` clean (61 files). Could not re-run the actual pipeline to confirm (heavy job, left to the user to trigger) -- the fix directly addresses the reported shape mismatch by construction (both arrays now provably come from the same grid), but should be confirmed against a real rebuild.

# 2026-08-07c: removed sea_mask_corrected.tif entirely -- native resolution was never actually required for flood-diagnostic masking (- JS)

User corrected the previous entry's assumption that `sea_mask_corrected.tif` legitimately needed to stay native resolution "to match subgrid-resolution flood output": "subgrid is only finer than coarse grid but aligned exactly the same way." Checked `src/postprocessing.py`'s three masking functions directly -- every one of them calls `da_sea.raster.reproject_like(da_dep_or_da_h, method="nearest")` at CONSUMPTION time, reprojecting whatever `sea_mask_path` they're given onto their own output grid (`compute_max_inundation`/`compute_flood_timeseries_stats`: the subgrid reference raster; `compute_flood_progression`: the coarse cell-resolution `zb`/`zs`). Subgrid is an exact integer subdivision of the coarse SFINCS grid, sharing its origin/axes -- reprojecting an already coarse, grid-aligned source onto it is a clean, well-defined upsample. The arbitrary-origin NATIVE FathomDEM/landuse pixel grid `sea_mask_corrected.tif` lived on was never actually required for that -- it was the same kind of unnecessary independent native resample already removed elsewhere in this refactor, just not recognized as such the first pass through.

**Fixed**: `sea_mask_corrected.tif` is removed entirely -- both its native-resolution construction in `10_depth_estimation_modelled.py` (`correct_sea_mask_from_grid_mask` call) and its passthrough copy in `10_depth_estimation_empirical.py`. `src.raster.correct_sea_mask_from_grid_mask()` is deleted (now unused everywhere). Rules `run_spinup`/`sanity_checks`/`run_event`/`compute_flood_metrics` (14/15/16/17) now point their `sea_mask` input directly at `zsini_sea_cells_on_grid.tif` (coarse, grid-aligned, weir-corrected -- the SAME file zsini itself is built from) instead. `10_depth_estimation_modelled.py`'s own internal per-round calibration diagnostics (`compute_max_inundation`/`compute_flood_progression` inside the round loop) previously used raw native `sea_mask.tif` too -- switched to a small coarse `ocean_mask_on_grid.tif` written once from `landuse_on_grid` (no `protected_pocket_mask` yet at that point in the round loop, since the weir isn't final -- round diagnostics never claimed weir-aware precision either). The native `sea_mask` input is now unused by both rule-10 siblings and removed from their own `.smk` inputs.

Verified: `tests/check_code_health.py` (61 files, no errors) and a full `snakemake -n` dry run from scratch through `compute_flood_metrics` for basin 2433835/scenario coast_100 resolves the entire DAG (20 rules) with no missing-input or ambiguous-rule errors, confirming `zsini_sea_cells_on_grid.tif` correctly wires into all four downstream flood-diagnostic rules.

# 2026-08-07b: one canonical coarse-grid landuse/roughness/sea classification, resampled exactly once -- no more per-consumer native-resolution resamples (- JS)

User flagged the zsini fix below as still architecturally inconsistent: `13_build_sfincs_skeleton.py` had ended up with TWO different construction methods for the same quantity (a native-resolution file-based pass, then an in-memory coarse override) -- "this is a mess... why do we not just use the same method... I think what i asked before is to map every, EVERY raster input data on the main coarse grid before any further use." Scope, per the user's own follow-up: landuse and everything downstream of it (roughness/friction, sea_mask, weir generation, zsini) -- NOT the DEM, which stays native resolution for subgrid.

**Root cause of the inconsistency**: native `landuse.tif` (rule `get_landuse`, 05b) was being independently resampled onto the SFINCS grid at THREE separate points -- `10_depth_estimation_modelled.py`'s own weir tracing (`GridArrays.sample_landuse`), `13_build_sfincs_skeleton.py`'s own weir-diagnostics section (a second, identical `sample_landuse` call), and HydroMT's own `roughness.create()` reprojection of native `roughness.tif` -- each a separate nearest-neighbor pass over the same source raster, capable of disagreeing with each other at the coastline fringe.

**Fixed**: new rule `grid_align_landuse` (09b, between `enforce_river_monotonicity`/09 and rule 10) resamples native `landuse.tif` onto the SFINCS grid (rule `build_sfincs_grid`'s own `sfincs_grid.json`) exactly ONCE, nearest-neighbor (categorical field, never averaged), producing `landuse_on_grid.tif`. Roughness is then built directly FROM this coarse classification via the same lookup-table reclassification `05c_get_roughness.py` uses on the native raster (`roughness_on_grid.tif`) -- never a separately-resampled continuous raster, so "this cell's roughness" and "this cell's land-use code" always agree by construction. Every downstream consumer of the landuse/sea/roughness classification now reads one of these two files directly, zero further reprojection: `modelled_depth_estimation`/`empirical_depth_estimation` (10, weir tracing + `zsini_sea_cells_on_grid.tif`, now built by BOTH depth_method siblings, not just "modelled"), and `build_sfincs_skeleton` (13, main-grid "manning" field + weir diagnostics). `GridArrays.sample_landuse()` (`src/protection_weir.py`) is removed -- no longer called anywhere.

Native-resolution `roughness.tif`/`landuse.tif` are NOT removed -- native roughness still feeds the SUBGRID table (`sf.subgrid.create`, both in rule 10's calibration model and rule 13's production build), which genuinely needs sub-cell detail for elevation AND roughness together, the same DEM exception the user asked to keep (two catalog entries now exist per script: `local_roughness`, coarse, main grid; `local_roughness_native`, native, subgrid only). `sea_mask_corrected.tif` also stays native -- it serves flood-diagnostic masking downstream (rules 14/15/16/17), matching subgrid-resolution flood *output*, a genuinely different resolution requirement than the SFINCS computational grid, not touched by this change.

`13_build_sfincs_skeleton.py`'s own zsini construction is also simplified as a side effect: since `zsini_sea_cells_on_grid.tif` is now unconditional (both depth_method modes produce it), the earlier native-resolution-pass-then-coarse-override split (2026-08-07 entry below) collapses into a single, unconditional construction -- one raster read, one connected-component fix, one `.create()` call, and the `zsini.tif` output FILE is now written directly from the same array that ends up in `sfincs.ini`, so a standalone inspection of the file always matches the simulation (previously the file lagged behind the in-memory override, which the user separately noticed as "thin stripes of no-data near the weir" in the on-disk file even though the actual `sfincs.ini` was already correct).

Verified: `tests/check_code_health.py` (61 files, no errors) and a `snakemake -n` dry run confirm `grid_align_landuse` schedules before `modelled_depth_estimation`/`build_sfincs_skeleton` and both correctly pick up `landuse_on_grid.tif`/`roughness_on_grid.tif` as new inputs.

# 2026-08-07: found and fixed the REAL zsini bug -- a double nearest-neighbor round-trip was silently un-doing ~53% of the weir correction (- JS)

User checked spin-up's own `sfincs.inp`, found it references a `sfincs.ini` still showing flooding behind dikes despite 2026-08-06h/i's fix, and asked what actually builds that file. Traced it: `13_build_sfincs_skeleton.py` builds `zsini`/`sfincs.ini` by reprojecting a NATIVE-resolution `sea_mask_corrected.tif` onto the SFINCS regular grid via HydroMT's own `reproject_like(method="nearest")`. That's a SECOND independent nearest-neighbor pass on top of `correct_sea_mask_from_grid_mask`'s own reprojection (coarse `protected_pocket_mask` -> native resolution, in rule `modelled_depth_estimation`). Verified empirically (after fixing a `round()`-vs-`floor()` pixel-indexing bug in the verification script itself, which had initially hidden the real result behind ~50% indexing noise): with the OLD approach, only 679/1,450 corrected fringe pixels (47%) actually ended up dry in the real, built `sfincs.ini` -- the rest were still wet, i.e. still flooding from the initial condition alone, exactly as the user reported. Two independent nearest-neighbor implementations don't invert each other cleanly, even on a grid confirmed pixel-identical between the two rules (`mmax=497, nmax=531, dx=dy=69.967, x0=289242.2, y0=4487731.966666667` -- checked directly against both rules' own `sfincs.inp`).

**Fixed**: `10_depth_estimation_modelled.py` now ALSO writes `{basin_id}_zsini_sea_cells_on_grid.tif` -- the real-open-sea classification (`ocean_mask & ~protected_pocket_mask`) written directly at THIS rule's own grid resolution, using `grid.transform`/`grid.crs` as-is, with NO reprojection at all (this is the exact same grid `13_build_sfincs_skeleton.py`'s own `sf.grid.data["dep"]` uses). `13_build_sfincs_skeleton.py` reads this file directly into an in-memory `xr.DataArray` built on `sf.grid.data["dep"]`'s own coords/transform/CRS, re-applies the same connected-component dry-out fix (now at coarse resolution, using the model's own `dep`/`mask`), and passes it to `sf.initial_conditions.create(ini=<DataArray>, reproj_method="nearest")` a SECOND time, right after the existing native-resolution call -- since source and destination grids are now identical, `reproject_like` is a confirmed true no-op (validated directly against the real skeleton model: 0/263,907 mismatched cells against a synthetic checkerboard test array), so this simply overwrites the previous, partially-inconsistent `ini` layer with a fully consistent one. Only applies in "modelled" mode (empirical mode has no weir/`protected_pocket_mask` at all, so its existing single-pass native reprojection was never at risk).

Re-verified after a full rebuild of both rules: all 1,450/1,450 targeted fringe pixels are now dry in the real, rebuilt `sfincs.ini` (was 679/1,450 before). `sea_mask_corrected.tif` (native resolution) is unaffected and still used for flood-diagnostic masking downstream (rules 14/15/16/17) -- a small residual resampling imprecision there affects only reporting/statistics, not the simulation's own physics, so it was left as a lower-priority concern relative to the zsini bug.

# 2026-08-06i: removed landuse_corrected.tif entirely -- sea_mask_corrected.tif already carried the exact same information, duplication spotted by user (- JS)

2026-08-06h added `landuse_corrected.tif` (for flood-diagnostic masking) alongside `sea_mask_corrected.tif` (for zsini), both derived from the same `protected_pocket_mask`. User pointed out this duplicates the same information: `src.postprocessing`'s three masking functions (`compute_max_inundation`, `compute_flood_progression`, `compute_flood_timeseries_stats`) only ever used `landuse_path` for one boolean check, `landuse.isin(WATER_LANDUSE_CODES)` i.e. `landuse==200` -- exactly what `sea_mask.tif` already encodes by construction (`sea_mask = lu_arr==200`, `05b_get_landuse.py`). Maintaining a second corrected raster just to re-derive the identical boolean was pure duplication.

**Fixed**: those three functions now take a `sea_mask_path` argument directly (`water_mask = da_sea_grid == 1.0`) instead of `landuse_path`/`WATER_LANDUSE_CODES`; the `WATER_LANDUSE_CODES` constant and the `water_landuse_codes` parameter are removed entirely (their whole role is now just "is sea_mask 1.0 here"). `landuse_corrected.tif` is removed as a rule 10 output (both `modelled_depth_estimation` and its `empirical_depth_estimation` sibling), along with `src.raster.relabel_landuse_from_grid_mask()` (now unused). Rules `run_spinup`/`sanity_checks`/`run_event`/`compute_flood_metrics` (14/15/16/17) now take a `sea_mask` input (`sea_mask_corrected.tif`) for the three masking functions, and their `landuse` input reverts to the RAW `landuse.tif` -- still needed, but only for two genuinely separate purposes unaffected by the weir correction either way: `compute_risk_metrics`'s `urban_code` (50) check, and `plot_inundation_check`'s cosmetic `water_bodies_path` background (checks `landuse==200` directly for drawing, unrelated to the stats masking).

`compute_risk_metrics` (urban exposure stats, rule `compute_flood_metrics` only) is unaffected and unchanged -- the 200->80 correction never touches code 50, so raw `landuse.tif` was already correct for that specific purpose. Verified against basin 2433835's real, already-computed `sea_mask_corrected.tif`/`sfincs_map.nc` (no pipeline rebuild needed, since the rule 10 output this reads was untouched by this refactor) -- all three refactored functions ran successfully end-to-end.

# 2026-08-06h: corrected root cause of 2026-08-06g -- grid-resolution mismatch, not isolated pockets; also fixed the real zsini/simulation bug, not just diagnostics (- JS)

2026-08-06g's own hypothesis (isolated landuse==200 components disconnected from the main ocean) turned out to be wrong for the reported basin (2433835): its landuse.tif/sea_mask.tif have exactly ONE connected sea component at native resolution -- confirmed by direct inspection, twice, after two rebuild-and-check cycles both came back with 0 relabeled pixels. Also ruled out, by reading `_pad_for_edge_tracing`/`seaward_edges` directly: a weir segment can only ever be drawn between a `land_mask` cell and an adjacent `water_like` cell (strict complements by construction), so a weir "cutting across" still-contiguous open water is structurally impossible in this algorithm -- ruling out a second hypothesis too.

**Actual root cause** (user's own diagnosis, from visually overlaying the weir on the landuse.tif): the weir is traced on `landuse_on_grid` -- the native landuse.tif (~30 m) resampled via NEAREST-NEIGHBOR onto the much coarser SFINCS regular grid (~70 m, `GridArrays.sample_landuse`). That resampling necessarily generalizes the true coastline to the coarse grid's own cell edges, so `land_mask` (the model's own "this is the protected/dry side" decision) disagrees with the fine native landuse.tif in a fringe running along the ENTIRE coastline -- not as isolated pockets, which is exactly why component-based detection kept finding nothing.

**Fixed**: `protected_pocket_mask` (`src/protection_weir.py`) is now `land_mask | separately_ringed_pocket` -- trusting the weir-building algorithm's own final `land_mask` directly as ground truth for "the protected side" (broad, essentially the whole inland footprint), instead of re-deriving connectivity. `relabel_landuse_from_grid_mask`'s own native-resolution `from_code==200` restriction naturally scopes this down to just the actual coastal mismatch fringe (an ordinary inland cell is never landuse==200 to begin with). `separately_ringed_pocket` (the original per-component check) is kept as an additional, narrower case: an ocean_mask component large enough to survive into `water_like` but not the main open-ocean component, walled off with its own ring while staying classified `water_like` throughout.

**Also fixed a second, more serious bug the user identified**: the original fix only touched `landuse_corrected.tif`, used by flood-diagnostic consumers (masking) -- but `13_build_sfincs_skeleton.py` builds the REAL production `zsini.tif` directly from `sea_mask.tif` (`sea_mask==1` -> initialize wet at `baseline_m`). An uncorrected fringe cell inside the protected area would start wet regardless of the weir, and SFINCS would then show it as "flooded" purely from its own initial condition (wherever `zsini > bed elevation`) -- a real simulation artifact, not just a masking/reporting issue. Added `correct_sea_mask_from_grid_mask()` (`src/raster.py`, same mechanism as `relabel_landuse_from_grid_mask` but clearing sea_mask's own `1.0 -> nodata` instead of relabeling a landuse code -- kept separate since `sea_mask.tif` also excludes nodata-elevation cells that `landuse.tif` doesn't know about). `10_depth_estimation_modelled.py` now writes `{basin_id}_sea_mask_corrected.tif` alongside `landuse_corrected.tif`; `13_build_sfincs_skeleton.smk`'s own `sea_mask` input now points at this corrected file instead of rule 05b's raw one -- this is the change that actually matters for the simulation. `10_depth_estimation_empirical.py` (builds no weir) passes both rasters through unchanged, same "unified filename, no depth_method-conditional file selection" convention as `landuse_corrected.tif`.

Verified via two new synthetic unit tests: a direct fine/coarse resampling scenario reproducing the exact resolution-mismatch mechanism (confirms `relabel_landuse_from_grid_mask`/`correct_sea_mask_from_grid_mask` correctly relabel/clear exactly the diagonal coastline fringe and leave deep land/ocean untouched), plus updated assertions on the original two-scenario test (large surviving pocket, small discarded-into-land patch) to match the new, intentionally broader `protected_pocket_mask` semantics.

# 2026-08-06g: relabel ocean pockets enclosed by the final coastal weir as permanent water body (landuse 80), not open sea (- JS) [root cause corrected by 2026-08-06h above]

User reported land behind the weir still showing landuse==200 (sea) in flood diagnostics. Root cause: `build_coastal_protection_weir()` classifies ALL landuse==200 cells as `water_like` and traces a protective ring around every large-enough connected component of it -- not just the real, open-ocean-connected one. A landuse==200 patch positioned inland (source-data imprecision, or a tidal pool caught in the same connected component as the coast) that's large enough to survive `min_component_cells` gets its own weir ring too, becoming a diked/enclosed pocket -- but the underlying landuse classification never changed, so it stayed "sea" for every downstream consumer (flood-depth masking, in particular).

**Fixed**: `build_coastal_protection_weir()` (`src/protection_weir.py`) now also computes `protected_pocket_mask` -- ocean-classified cells that survive into `water_like` but belong to a DIFFERENT connected component than the largest one (the real open ocean, same "single largest component" convention already used by the skeleton's own isolated-sea-pocket zsini fix). `10_depth_estimation_modelled.py` uses this (via a new `src.raster.relabel_landuse_from_grid_mask` helper) to write a NEW output, `{basin_id}_landuse_corrected.tif`: a copy of the native-resolution landuse.tif with these specific pixels relabeled 200 → 80 (permanent water body). Deliberately a SEPARATE file, not an in-place edit -- landuse.tif stays the untouched ground truth for roughness, the weir algorithm's own `ocean_mask` on any future rerun (relabeling in place would create a feedback loop: `build_coastal_protection_weir` treats landuse==80 as plain land, so the ring around a relabeled pocket would silently vanish on the very next rerun), and the skeleton's own weir-diagnostic plot (shows the original classification the weir was actually traced against).

Rules `run_spinup`, `sanity_checks`, `run_event`, and `compute_flood_metrics` (14/15/16/17) now read `landuse_corrected.tif` instead of the raw `landuse.tif` for their own flood-depth masking (`src.postprocessing.WATER_LANDUSE_CODES`) -- since that constant is deliberately `(200,)` only, not `(80, 200)` (see the earlier, explicit `[[project_water_landuse_codes_bug]]` decision to keep inland water/rivers visible in flood stats), relabeled pockets now show up as meaningful flood-depth data instead of being blanked out as featureless open sea. `sea_mask.tif`/zsini are UNAFFECTED (still derived from the raw, uncorrected landuse.tif) -- these pockets are still permanent water physically, so they should still initialize wet at `baseline_m`; only their classification LABEL for diagnostic-masking purposes changes. Manning's n for codes 80 and 200 are numerically identical in `lu_to_roughness_lookup.csv` (0.02 both), so roughness.tif needs no corresponding fix. `10_depth_estimation_empirical.py` (the sibling rule for `depth_method == "empirical"`, which never builds a coastal weir) writes `landuse_corrected.tif` too, as an unchanged passthrough copy of `landuse.tif` -- same "both siblings write the same canonical filename" convention already used for `river_burned_dem`/`coastal_protection_weir`, so downstream rules never need depth_method-conditional file selection.

# 2026-08-06f: fixed 13_build_sfincs.py's own "05_forcing.png" diagnostic to actually reflect slr_m/discharge_multiplier (- JS)

Missed on the first pass: `13_build_sfincs.py` rebuilds the surge/river forcing a SECOND time, independently, purely to plot "what was really built" for this scenario (`05_forcing.png` -- its own comment says so explicitly, mirroring the real build "RP-for-RP and mode-for-mode"). That second call site (`build_design_surge_matrix(_sds, design_rp_surge_yr)` / `build_design_discharge_matrix(_rds, _active_mask, design_rp_river_yr, apply_protection_floor=...)`, ~line 477/485) never got the `slr_m=effective_slr_m`/`discharge_multiplier=discharge_multiplier` arguments added alongside the main build calls -- so the plot would have silently shown the UNSCALED forcing while `sfincs.bzs`/`sfincs.dis` actually contained the scaled one. Fixed by passing both through to this second call site too. Rule 07's own preview call (`07_get_boundary_forcings.py`, fixed diagnostic RP, explicitly basin-level/illustrative-only) is correctly unaffected by design -- it never reads either factor, exactly like the calibration/skeleton/spin-up.

# 2026-08-06e: added boundary_forcings.river.discharge_multiplier -- a deferred, build-time scaling factor on river discharge (- JS)

Following the same pattern just established for surge.slr.slr_m: added `discharge_multiplier` (default 1.0), a uniform scaling factor applied to the built discharge hydrograph at every active river seed/boundary crossing. `src.river_forcing.build_design_discharge_matrix()` gained a `discharge_multiplier` argument, multiplying the final hydrograph (bankfull lead-in and event peak alike -- equivalent to scaling both endpoints before building the sinusoid, since the wave shape is affine). Applied ONLY in `13_build_sfincs.py` (which reads it as its own param, `boundary_forcings.river.discharge_multiplier`), never baked into `river_forcing.nc` and never passed to rule 10's weir/depth calibration (`build_calibration_seed_discharge`, a separate function) or rule `run_spinup` (`interpolate_discharge_at_rp`, also separate) -- both remain naturally unaffected since they don't call `build_design_discharge_matrix` at all. Also fixed the compound-lag shift's own bankfull padding value (previously read `river_ds.bankfull_discharge` directly, unmultiplied) to scale by the same factor, keeping the padding consistent with the now-multiplied discharge data it pads.

# 2026-08-06d: SLR fingerprint application deferred to scenario-build time -- surge_forcing.nc no longer depends on the slr_m target (- JS)

Changing `boundary_forcings.surge.slr.slr_m` was forcing a full rerun of rule 10's expensive weir/depth calibration and rule `build_sfincs_skeleton`, even though neither actually uses the storm-surge magnitude (only the MDT-only `baseline_m` for zsini). Root cause: `apply_slr_fingerprint()` used to scale the AR6 fingerprint by `slr_m` immediately and bake the result into `rp_level`/`station_baseline`, which fed into `surge_forcing.nc`'s scalar `baseline_m` too -- and Snakemake invalidates every declared consumer of a file on ANY change to it, regardless of which field inside actually changed.

**Fixed per explicit direction**: `apply_slr_fingerprint()` now only computes and stores the dimensionless `slr_fingerprint` ratio (local/global-mean SLR, depends only on `ssp_scenario`/`confidence_level`/`year`/`quantile`); it no longer touches `rp_level` or takes `slr_m` at all. `07_boundary_forcings.smk`'s own `surge_slr` param dict now excludes `slr_m` entirely, so rule `get_boundary_forcings` (and therefore `surge_forcing.nc`'s mtime) is completely independent of the slr_m config value. `lookup_storm_tide_at_rp()`/`build_design_surge_matrix()` (`src/surge.py`) gained an optional `slr_m` argument, applying `slr_fingerprint * slr_m` on top of the MDT-only levels at call time; `13_build_sfincs.py` reads `slr_m`/`slr_enabled` as its OWN params and passes them through -- so changing slr_m now only reruns the cheap per-scenario `build_sfincs` + its downstream event run, never rule 07, rule 10, or the skeleton build. `14_run_spinup.py` deliberately does NOT pass `slr_m` (stays MDT-only) -- SLR applies only to the actual named-scenario's own forcing, not shared basin-level infrastructure; the event's own lead-in absorbs the resulting handoff transient, same as it already does for the MDT-only zsini. Updated `plots.py`'s two rule-07 diagnostics (`plot_surge_corrections`, the forcing-timeseries per-station table) to show the fingerprint ratio instead of a baked-in SLR value, and `tests/plot_surge_vs_dike_crest.py` (a standalone, non-Snakemake script) to read the config's slr_m target itself and apply it explicitly, matching production.

# 2026-08-06c: removed the flood-stability half of the calibration's early-stop check -- containment alone now converges it (- JS)

Basin 2433835's river-depth calibration was burning through every one of its `n_correction_iterations` (10) even though the log showed zero overtopping (`0 badly overtopped, 0 near target`, centerline + coastal + river-boundary probes) from round 4 onward. Traced it to the early-stop condition in `10_depth_estimation_modelled.py`, which required BOTH: (a) `contained` (no overtopping anywhere -- the part that was actually satisfied), AND (b) `flood_stable` (the realized inundated-cell count within 10% of round 0's own count). Round 0 is confined by an effectively un-overtoppable 1000 m wall, so its own flooded footprint is artificially tiny (266,009 cells for this basin) -- the real, correctly-contained crest settled into its own genuine, stable steady state at 695,312 cells (2.61x round 0's) from round 3 onward and never got within 10% of the artificial baseline, so `flood_stable` never fired despite `contained` being true for many rounds running.

**Fixed per explicit direction**: removed the `flood_stable` condition and the `current_inundated_count`/`round0_inundated_count` tracking entirely. Convergence now fires on `contained` alone -- physically the right criterion (no overtopping = nothing left to correct), and inspecting for overtopping directly is exactly what this was for. Updated the loop's own comments and `config.yml`'s `n_correction_iterations` description to match; noted why the removed check existed and why it was overly protective, so a future session doesn't reintroduce it by reflex.

# 2026-08-06b: OSM land polygons removed from the pipeline entirely -- land_polygons.gpkg (rule 03) is now vectorized straight from landuse, and flood diagnostics mask out real sea (- JS)

Follow-up to the sea_mask fix earlier today: per explicit direction, OSM is no longer used anywhere as a "what counts as land" source, project-wide.

**`WATER_LANDUSE_CODES`** (`postprocessing.py`): changed from `(0, 2000)` (a deliberate no-op chosen in an earlier session so ocean/rivers stayed visible in flood diagnostics) to `(200,)` (the real sea code). `compute_max_inundation`, `compute_flood_timeseries_stats`, and `compute_flood_progression` now actually exclude open sea from flood-extent depth/area/volume stats and from `plot_max_inundation_map`/`animate_flood_progression`/`plot_inundation_check`. This intentionally reverses the earlier "don't touch, ocean stays unmasked on purpose" decision -- noted in memory in case a future session sees the old value referenced anywhere.

**`land_polygons.gpkg`** (rule `get_land_polygons`, 03): previously clipped the OSM land-polygons dataset to the domain bbox. Now vectorizes `landuse != 200` from the same global Copernicus landuse raster `get_landuse` (05b) already uses, native WGS84, via a new shared helper `src.raster.vectorize_land_from_landuse`. The rule's name, output key, and output file path are all unchanged, so **every one of its ~15 downstream consumers needed zero code changes** -- every diagnostic plot's background overlay, and the two real, functional `exclude_polygon` uses in `10_depth_estimation_modelled.py`/`13_build_sfincs_skeleton.py`'s own waterlevel boundary mask, all automatically pick up the landuse-derived file. This also makes the boundary mask agree with `sea_mask.tif` and `src.protection_weir`'s own `ocean_mask` -- all three now trace back to the exact same `landuse == 200` criterion.

**`get_protection_levels`** (rule 04) had its own, separate, independent OSM read (not via rule 03 -- its docstring explicitly says it depends only on the delta polygon, not the model-domain chain, so it can't just depend on rule 03's per-basin output either). Fixed the same way: vectorizes its own small window directly from the raw landuse catalogue source via the same shared helper, written to a throwaway temp file (never a declared Snakemake output) since `plot_protection_levels` expects a file path.

**Removed**: the `osm_land` data-catalogue entry (`config/data_catalogue.yml`) -- no longer referenced anywhere.

Verified via `tests/check_code_health.py` and `snakemake -n all` (single basin/scenario): `get_land_polygons`/`get_protection_levels` correctly show as needing re-run (their own logic changed); everything downstream cascades normally with no DAG breakage.

# 2026-08-06: sea_mask.tif is now landuse==200 only, no OSM land polygons -- fixes "already wet at t=0 with no weir" on land the weir never walled off (- JS)

Debugging a scenario where land was already flooded at the very start of the run (before any real forcing had time to act), even though the coastal weir looked correctly positioned and zsini looked fine in isolation: traced `sea_mask.tif` (rule get_landuse, 05b) and found it was built as `~land_mask | (lu_arr == 200)` -- sea wherever OSM's own land polygon didn't cover the cell, OR wherever landuse said 200, whichever was more generous. Meanwhile `src.protection_weir`'s own `ocean_mask` (used to decide where the coastal protection weir actually gets traced, in `10_depth_estimation_modelled.py`, `13_build_sfincs_skeleton.py`, and `protection_weir.py` itself) has only ever checked `landuse == 200`, nothing else.

Quantified the disagreement on basin 2433835: of all pixels `sea_mask` called sea, 3.9% (12,898 px) had a landuse code other than 200 -- 87% of those were code 80 ("inland water" in the Copernicus LC100 scheme, i.e. tidal flats/lagoons), and OSM's own coastline data was the reason `sea_mask` still called them sea despite the differing landuse code. Net effect: `zsini` (built from `sea_mask`) started those cells wet, correctly reflecting OSM's view -- but the weir (built from the narrower `landuse==200` check) never walled them off, since it disagreed with OSM about whether they were water at all. Already-flooded land at t=0, no barrier standing against it.

**Fixed per explicit direction**: `sea_mask.tif` is now `landuse == 200` alone -- the OSM land-polygon term is dropped entirely (`05b_get_landuse.py`). This makes `zsini` and the weir's `ocean_mask` agree by construction: both now read the exact same landuse criterion, so they can't drift apart again. `land_polygons` stays as a rule input, but only for the diagnostic plots' own overlay (`plot_landuse`/`plot_sea_mask`), not the classification itself.

# 2026-08-05b: rule 13's own forcing-timeseries plot always showed a fixed preview RP, never the scenario's actual surge_rp -- fixed, and removed the redundant config constant that caused it (- JS)

While debugging a "surprisingly little flooding at surge_rp=500" report, traced `boundary_forcings.surge.return_period` (config.yml): it's a basin-level, scenario-independent diagnostic-preview RP (the surge-side twin of the already-documented `boundary_forcings.river.eva.rp_fl`) used only for rule 07's own `rp_level`/`rp_level_raw` preview columns and `07_surge_correction.png` -- it was never used anywhere near `baseline_m`/`coastal_protection_crest_m` or any other production quantity (confirmed by tracing the actual math: `baseline_m = mean(-mdt + slr_m)`, which cancels out the RP entirely).

**Real bug found along the way**: `13_build_sfincs.py`'s own forcing-timeseries diagnostic plot (`05_forcing.png`) read `surge_forcing.nc`'s stored `water_level` field directly -- built at that same fixed preview RP -- instead of rebuilding it at the scenario's own `design_rp_surge_yr`. The river panel of the same plot already correctly used `design_rp_river_yr`; the surge panel did not. So `05_forcing.png` has been silently showing the wrong surge curve for every scenario whose `surge_rp` differs from the config's fixed preview value -- never what actually went into that scenario's own `sfincs.bzs` (which correctly used `build_design_surge_matrix(surge_ds, design_rp_surge_yr)` elsewhere in the same script all along).

**Fixed**:
- `13_build_sfincs.py`: the diagnostic plot now rebuilds the surge matrix via `build_design_surge_matrix(surge_ds, design_rp_surge_yr)` (flat `river_only_flat_level_m` for `forcing_mode="river_only"`), mirroring the river panel exactly. Plot title now labels which RP (or flat mode) is actually shown.
- `boundary_forcings.surge.return_period` removed from `config.yml`. Rule `get_boundary_forcings` (07) now sources its own diagnostic-preview RP from the `"default"` scenario's own `surge_rp` (`config/scenarios.yml`) instead of a separate, redundant constant -- one source of truth instead of two numbers that can silently drift apart. Still fully basin-level and independent of `target_scenarios` (always reads `scenario_params("default")`, never whichever scenario is actually requested), preserving the exact no-recoupling property `rp_fl` was designed for. Added a fail-fast check in `07_boundary_forcings.smk` if `"default"` itself has `surge_rp: null`.
- **Also found and fixed**: `config/scenarios.yml`'s `"default"` scenario had `surge_rp: 200`, not a COAST-RP tabulated value (`1, 2, 5, 10, 25, 50, 100, 250, 500, 1000`) -- this broke `00_common.smk`'s own validation and blocked every single snakemake invocation, unrelated to the above. Reset to `250` (nearest tabulated value) per user direction.

Verified via `tests/check_code_health.py` and a `snakemake -n build` dry run (3 scenarios): `get_boundary_forcings` still runs once per basin, `build_sfincs` once per scenario, unchanged from before.

# 2026-08-05: fixed NoDataException in build_sfincs/run_spinup -- sf.config's tref/tstart/tstop were never set before calling water_level.create()/discharge_points.create() (- JS)

A colleague's first real end-to-end run hit `hydromt.error.NoDataException: DataFrame has no data after time slicing.` inside `sf.water_level.create()` in `14_run_spinup.py`.

**Root cause**: `water_level.create()`/`discharge_points.create()` internally call `self.model.get_model_time()`, which reads `tstart`/`tstop` straight off the in-memory `sf.config` (a hydromt_sfincs Pydantic model), then time-slices the passed-in forcing DataFrame against that range. Both `13_build_sfincs.py` and `14_run_spinup.py` load their basin's skeleton read-only (`13_build_sfincs_skeleton.py`'s own output), which deliberately never writes `tref`/`tstart`/`tstop` into its `sfincs.inp` (the skeleton isn't meant to be runnable on its own) -- so `sf.config` still carried hydromt_sfincs's own class defaults (today's date, `datetime.now()`) at the moment `.create()` ran. Both scripts only wrote the *real* `tref`/`tstart`/`tstop` afterward, into a hand-crafted `sfincs.inp` text file (plain file I/O, bypassing `sf.config.write()`) -- so the in-memory model's own time window never matched the forcing timeseries actually being passed in (indexed at the pipeline's real `tref`, e.g. 2000-01-01), and the slice came back empty. `10_depth_estimation_modelled.py` already had the correct pattern (`sf.config.set` for `tref`/`tstart`/`tstop` right after computing them, with a comment noting it should match `13_build_sfincs.py`) -- the skeleton-split rewrite of `13_build_sfincs.py`/`14_run_spinup.py` earlier this session dropped that step.

**Fixed**: added `sf.config.set("tref", tref)` / `sf.config.set("tstart", tref)` / `sf.config.set("tstop", tstop)` right after computing `tref`/`tstop` in both scripts, before any `.create()` call. No effect on the actual `sfincs.inp` written to disk (still hand-crafted, unchanged) -- this only fixes what the in-memory `.create()` calls see.

# 2026-08-04d: removed quadtree grid support entirely -- regular grid only from now on (- JS)

The project will not use quadtree grids, so every quadtree-specific code path was removed rather than left as dead, never-enabled config.

**Removed**: `sfincs.grid.quadtree` config block (and the now-orphaned `weir_crest_junction_blend_m` key, whose only caller was the quadtree weir fallback below) from `config.yml`; the `quadtree.enabled requires subgrid.enabled` validation in `00_common.smk`; `src/quadtree_refinement.py` (whole file, `build_refinement_polygons`); `GridArrays.from_quadtree`/`weighted_graph`/`_seaward_edges_quadtree(_with_values)` and the `grid_type` dispatch in `src/protection_weir.py` (now regular-grid only); `build_channel_mask_quadtree`/`build_smoothed_weir_crest_quadtree`/`_quadtree_face_polygons` in `src/river_burn.py`; `plot_refinement_zones`/`_mesh_overlay_setup` and the mesh-native (`xu.UgridDataArray`) branch of `animate_flood_progression` in `src/plots.py`; `_mosaic_quadtree_dep_levels` and the `UgridDataArray` branches of `compute_flood_progression` in `src/postprocessing.py` (`get_bed_level` now only reads `dep_subgrid.tif`); the `elif depth_method == "modelled" and quadtree_enabled:` weir-building branch and the post-write `ncinifile`/`inifile` workaround in `13_build_sfincs_skeleton.py`; the `quadtree_enabled`-conditional centerline-snapping branches in `13_build_sfincs.py`/`14_run_spinup.py` (now unconditional); the `refinement_polygons.gpkg`/`01b_refinement_zones.png` conditional outputs in `13_build_sfincs_skeleton.smk` and the `Snakefile`; and the quadtree-only `dep_subgrid_lev*.tif` handling in `tests/test_subgrid_river_depth_profile.py`/`tests/test_bank_elevation_check_sfincs.py`/`tests/plot_main_grid.py`.

**Not removed** (flagged instead, since it's a separate, pre-existing dead-code question outside this scope): `build_smoothed_weir_crest_regular` (+ its helper `_smoothed_weir_crest_profiles`) in `river_burn.py` now has zero remaining callers -- its only caller was the removed quadtree fallback. Left in place for the user to decide on separately.

Verified via `tests/check_code_health.py` and a `snakemake -n` dry run.

# 2026-08-04c: fixed a stale "elevation_merged" name in build_sfincs_skeleton that actually meant elevation_conditioned (- JS)

While confirming grid/subgrid generation is handled identically for both `depth_method` modes, found that `build_sfincs_skeleton`'s own input key, Python variable, and data-catalog key were all named `elevation_merged`/`local_elevation_merged` -- but the rule's own input mapping had always pointed that key at `elevation_conditioned.tif` (post-monotonicity), never the actual raw `elevation_merged.tif` (rule 05a's output). This predates today's skeleton split -- it was already this way in the original monolithic `13_build_sfincs.py`, just carried forward faithfully. Initially misread this as a real raw-vs-conditioned inconsistency between the skeleton's own subgrid fallback and `10_depth_estimation_modelled.py`'s own subgrid fallback (which correctly uses a key named `local_elevation_conditioned`) -- they were already reading the identical file, just under a confusing name.

**Fixed**: renamed the input key (`elevation_merged` → `elevation_conditioned`), the Python variable (`elevation_merged_path` → `elevation_conditioned_path`), and the data-catalog key (`local_elevation_merged` → `local_elevation_conditioned`) throughout `13_build_sfincs_skeleton.smk`/`.py` to say what they've always actually been. No behavior change -- `elevation_list_subgrid`'s fallback source is the same file as before, just correctly named now, and now visibly matches `modelled_depth_estimation`'s own convention. Also corrected two `Reference_memory.txt` passages that had independently inherited the same stale naming (one claimed a 3-level elevation fallback for the main grid -- burned > conditioned > raw merged -- that was never real; the code has always been a 2-level fallback, burned > conditioned only).

# 2026-08-04b: decoupled spin-up from scenario RP changes -- split rule build_sfincs into a basin-level skeleton + per-scenario forcing (- JS)

Changing a scenario's own design RP (surge_rp/river_rp in config/scenarios.yml) used to force rule `run_spinup` to re-run too, even though spin-up physically doesn't depend on the event's design RP at all -- it depended on that scenario's own `sfincs.inp`, which rule `build_sfincs` rewrote in full (grid/elevation/mask/weir/roughness/subgrid/forcing, one atomic `sf.write()`) on every RP change.

**Root cause**: traced through the whole of `13_build_sfincs.py` and confirmed most of it (grid, elevation, mask, weir, roughness, subgrid, observation points, simulation timing) is scenario-INDEPENDENT -- only initial conditions (forcing_mode-dependent), water-level/discharge forcing, and `rstfile`/`tstart` actually vary by scenario. But since the whole model was written as one atomic `sf.write()`, even the independent files got rewritten (mtime touched) on every RP change, which is enough to invalidate any downstream Snakemake rule depending on them regardless of which specific file it declared as input.

**Fixed** by splitting rule `build_sfincs` into two:
- **`build_sfincs_skeleton`** (new, basin-level, no `{scenario}` wildcard): builds everything scenario-independent, writes to a new `results/{basin_id}/sfincs_skeleton/`.
- **`build_sfincs`** (per-scenario, much smaller now): loads the skeleton read-only, redirects writes to `runs/{scenario}/sfincs/` (`sf.root.set(...)`, the same pattern already used in `10_depth_estimation_modelled.py`'s own calibration round loop), builds only the forcing components, writes ONLY those (`sf.water_level.write()`/`sf.discharge_points.write()` -- confirmed independently callable per-component writers), and hand-crafts this scenario's own `sfincs.inp`: the skeleton's grid-header and geometry `*file` entries are forwarded via a relative path (new `src.sfincs_run.parse_sfincs_inp`/`forward_geometry_files` helpers, same technique `14_run_spinup.py` already used for borrowing from the old per-scenario build). HydroMT's own config-writing path (`get_set_file_variable`) silently absolutizes any file reference outside the model's current root instead of preserving a relative path, so this genuinely can't be done via `sf.write()`/`sf.config.write()` -- confirmed via a scoped investigation of the `hydromt_sfincs` source.

Rule `run_spinup` was rewritten the same way, and moved to basin-level (`results/{basin_id}/spin_up/`, confirmed with the user -- since it's now provably identical for every scenario of a basin, running it once per scenario would just be redundant SFINCS execution): it always runs at a fixed **RP=1** (coast AND river -- both exactly tabulated in COAST-RP and the river GPD return-value table respectively, no extrapolation needed), held CONSTANT over `spinup_days` (a flat 2-point timeseries, not a sinusoidal ramp), built via the existing `lookup_storm_tide_at_rp`/`interpolate_discharge_at_rp` lookups -- no longer sliced from any scenario's own event hydrograph. Rule `sanity_checks` moved to basin-level too (it only ever analyses spin-up's own output).

**Side fix found along the way**: `compute_max_inundation`/`compute_flood_timeseries_stats` (rules `run_event`, `compute_flood_metrics`) take a separate "subgrid lookup root" argument -- previously always the scenario's own `sfincs_root` (correct, since subgrid lived there). Now that the subgrid reference raster physically lives only in `sfincs_skeleton/`, both call sites needed a new `skeleton_root` param; without this fix they would have silently fallen back to the coarser cell/mesh-resolution bed level instead of raising.

Verified via `snakemake -n` with two scenarios (`coast_100`, `river_100`): `build_sfincs_skeleton`/`run_spinup`/`sanity_checks` each appear exactly once in the job stats, while `build_sfincs`/`run_event`/`compute_flood_metrics` appear once per scenario -- confirming the decoupling actually works, not just in theory.

# 2026-08-04: reorganized results/{basin_id}/ folder structure (- JS)

Preprocessing visuals lived in a separate top-level `visuals/input_data/` tree instead of next to the `inputs/` they document, calibration lived in a bare `sfincs_calibration/` instead of being grouped with the other preprocessing inputs, and spin-up's own inputs/outputs/visuals were split across two different subtrees nested inside each scenario's own `sfincs/`/`visuals/` folders. Reorganized to:

```
results/{basin_id}/
├── preprocessing_inputs/               (renamed from inputs/)
│   ├── domain/                         (unchanged)
│   ├── forcing/                        (unchanged)
│   ├── visuals/                        (moved from top-level visuals/input_data/, "input_data" segment dropped)
│   └── depth_crest_calibration/        (renamed+moved from top-level sfincs_calibration/)
└── runs/{scenario}/                    (renamed from scenarios/{scenario}/)
    ├── sfincs/                         (build outputs -- unchanged, except rstfile now points sideways to ../spinup/)
    ├── spinup/                         (NEW sibling of sfincs/ -- spin-up's own sfincs.inp, referencing
    │                                     ../sfincs/<file> instead of today's "../<file>"; its own
    │                                     rst/map.nc/his.nc outputs; and its own visuals, all in one
    │                                     place instead of split across sfincs/spinup/ and visuals/spinup/)
    ├── visuals/                        (unchanged, except no nested spinup/ subfolder anymore)
    └── metrics/
```

Existing on-disk results are NOT migrated -- old folders are left as-is (or deleted manually later); everything rebuilds fresh under the new layout on the next run.

**`sfincs_grid/` empty-folder fix**: `rule build_sfincs_grid` (08c) used to instantiate a throwaway `SfincsModel(root=..., mode="w+")` purely to call `create_from_region()` and read back the grid's transform/shape -- `sf.write()` was never called, but HydroMT still created the root directory on init, leaving a permanently empty `{basin_id}/sfincs_grid/` folder. Investigated whether the rule could be removed entirely (its only consumer, rule `enforce_river_monotonicity`/09, deliberately has no hydromt import -- a plain rasterio/Affine consumer per 08c's own docstring, so inlining would add a new dependency there). Kept the rule, but rewrote its implementation to call `hydromt.model.processes.create_grid_from_region()` directly -- the same standalone function `SfincsModel.grid.create_from_region()` delegates to internally -- instead of instantiating a `SfincsModel`. No Model root directory is ever created now, not just cleaned up after the fact.

Every `.smk` rule file, the `Snakefile`'s own output lists, several script docstrings, two standalone `tests/*.py` dev scripts, and `Reference_memory.txt`'s folder-tree documentation were updated to match. Three folders (`visuals/model_runs/`, `sfincs/validation_higher_rp/`, `sfincs/validation_protection_level/`) are confirmed stale leftovers from rule 13b (removed 2026-08-02) that no code regenerates -- left for manual deletion whenever convenient.

# 2026-08-03c: updated surge diagnostic plots to stop showing protection as part of the boundary correction chain (- JS)

Follow-up to the previous entry's `surge.py` fix (protection level no longer subtracted from the boundary water level): two diagnostic plots still described/displayed that subtraction as if it still happened, which would now be actively misleading.

`plot_forcing_timeseries` (`src/plots.py`): the surge (left) panel's dead `surge_corrected` branch (nothing ever set `water_level_uncorrected` on `surge_ds`, confirmed by grep -- always took the `else` path) drew a "Protection level" reference line and an original-vs-corrected legend that could never actually appear; removed entirely, left panel now always plots the single effective series. The per-station correction table dropped its `−prot`/`final_peak` columns and the `baseline_m` row's label, both of which never matched the netCDF's own `baseline_m` (that was always `mean(−MDT+SLR)`, no protection term, even before the `surge.py` fix). River-side `modify_hydrograph` discharge-correction plotting (a real, still-active, opt-in feature, unrelated to the coastal fix) is untouched.

`plot_surge_corrections` (`src/plots.py`, the bar-chart diagnostic): dropped the `protection_level_raw` parameter and its `−protection` bar entirely; `net_peak` is now `rp_raw − MDT + SLR` (was `− prot` too). Updated `07_get_boundary_forcings.py`'s call site to match (no longer passes `protection_level_raw`).

Also fixed a stale comment in `13_build_sfincs.py` (`baseline_m significantly negative... after protection-level correction`) -- `baseline_m` never included a protection term, even before this session's fixes.

# 2026-08-03b: fixed the coastal boundary double-counting FLOPROS protection (subtracted from water level AND used as weir crest) (- JS)

`src/surge.py`'s `build_design_surge_matrix` (called from `13_build_sfincs.py` to build the actual `.bzs` water-level boundary for every scenario build) subtracted `surge_ds["protection_level"]` from the wave timeseries. That `protection_level` is the SAME `mean_prot_raw` value (RP41.2 FLOPROS coastal standard, pre-MDT-shift) that `07_get_boundary_forcings.py` also uses, one line later, to derive `coastal_protection_crest_m` -- the weir crest baked into the model at rule 13. One computed value, two independent consumers: the boundary forcing was reduced by the same protection standard the weir was already floored at, suppressing the flood signal twice for the same defense.

This directly contradicted `07_get_boundary_forcings.py`'s own design comment at the point `protection_level` is computed ("water_level itself is never touched here ... a weir provides a real barrier, unlike subtracting a scalar protection height directly from the water_level boundary forcing"): rule 07 already got this right for its OWN diagnostic `water_level` field (`build_surge_dataset` never subtracts it); the bug was that `build_design_surge_matrix`, called later at rule 13's actual build time for the scenario's real design RP, re-implemented the subtraction anyway.

**Fixed**: removed the subtraction from `build_design_surge_matrix` entirely. `protection_level` is still stored on `surge_forcing.nc` (used by `coastal_protection_crest_m` and the diagnostic plot) -- it's just no longer subtracted from the boundary a second time. Every coastal/compound scenario's actual `.bzs` water-level boundary will now sit `~mean_prot_raw` higher than before (e.g. +0.4467 m at basin 2433835) -- a real, intentional change in forcing magnitude, not just a bugfix nuance; expect more coastal flooding across the board on any basin's next rebuild.

Not yet re-run anywhere -- this changes the actual boundary forcing for every basin using `depth_method` regardless of coastal/compound/river-only mode split (any scenario with a non-null `surge_rp`), so every affected basin's `build_sfincs`/scenario run needs a rebuild to pick this up, not just basin 2433835.

# 2026-08-03: fixed _rasterize_nearest's unbounded propagation, the actual cause of anomalously high coastal/river-boundary crest (- JS)

Colleague reported basin `2433835` (Ebro Delta) showing no visible difference in flooding between increasingly severe coastal scenarios. Investigation of the basin's own `sfincs_build` weir visuals found calibrated coastal weir crest anomalously high (median ~4.8 m, up to ~7-13 m) across much of the open coast, far from any river influence -- clearly wrong for a flat, low-lying delta coast.

**First fix (real bug, but NOT the cause of the reported symptom)**: `10_depth_estimation_modelled.py`'s one-shot "snap" step (the final tightening once a round converges) directly overwrote `coastal_crest_current`/`river_boundary_crest_current` with `np.maximum(period_max_zs + freeboard_m, coastal_protection_crest_m)`, with no NaN guard -- unlike the regular per-round update just above it (which already correctly preserves the existing tracked value via `np.where` when a probe's water level is NaN that round). A probe cell that happened to be dry in the specific round the snap fired got permanently poisoned to NaN, with no later round ever able to recover it. Fixed the same way the regular update already does it (`np.where(np.isfinite(...), target, current)`), for both coastal and river-boundary probes. Also restricted coastal probe cell *selection* to the model's own active mask (`grid.valid_mask`) -- 5.7% of the exact probe cells fell outside the active domain and were therefore permanently unsimulated/NaN by construction.

Verified via an offline replay against basin `2433835`'s own already-existing round1-3 `sfincs_map.nc` files (no new SFINCS runs) that this fix, on its own, produces a BYTE-IDENTICAL final weir gpkg to the unfixed code for this basin -- the specific probe cells this bug poisons happen to sit where `river_crest_on_grid` already dominates via `max()`. Real bug, worth fixing, but not the explanation for the reported symptom.

**Second fix (the actual cause)**: `_rasterize_nearest` (used to paint both `coastal_crest_current` and `river_boundary_crest_current` onto the full grid) had NO distance cutoff at all -- every cell in the entire domain took the value of its nearest probe, however far away. `river_boundary_probe`'s own probes sit right next to the channel/discharge points by construction, and one of them legitimately computed a high crest (up to ~12.6 m in the replay) near what is very likely a real hydraulic bottleneck at a discharge injection point (a known, documented scenario -- see the probe setup's own docstring). With no cutoff, that single local reading propagated across the ENTIRE coastline: checking cells more than 50 grid cells from any river channel cell in the replay showed the SAME inflated crest (min 3.0 m, median 5.5 m, max 8.0 m) purely from unbounded nearest-neighbour propagation, not any genuine local water level.

**Fixed**: `_rasterize_nearest` now takes a required `max_distance_cells` argument; cells beyond that distance get NaN instead of inheriting a possibly-kilometres-away probe's value, and `build_coastal_protection_weir`'s own existing NaN fallback (to the flat `coastal_protection_crest_m` floor) takes over from there -- mirroring how `river_crest_on_grid` already uses a *bounded* `grid.dilate_values(...)` rather than an unbounded fill. Capped at `river_crest_dilation_cells` (same radius already used for the river crest's own reach onto land), at all four call sites (per-round build, final re-trace).

Verified via the same offline replay: combining this with the NaN-guard fix drops the reconstructed weir's median elevation from 5.5 m to 0.5 m (the flat floor) and its max from 12.6 m to 7.0 m -- the snap-guard fix alone changed nothing, confirming the distance cutoff is what actually resolves the reported symptom.

Not yet done: basin `2433835`'s production model has not been re-run end-to-end with this fix (or the discharge ramp / crest redesign fixes from 2026-07-29) -- the replay above is an offline reconstruction from existing round1-3 data (and a simplified river-boundary probe set, discharge-buffer probes omitted), not a full re-simulation. A fresh `modelled_depth_estimation` + `build_sfincs` re-run is still needed to confirm end-to-end.

Separately flagged, not fixed: `src/surge.py`'s `build_design_surge_matrix` subtracts `protection_level` from the surge boundary water-level timeseries, contradicting its own documented invariant and rule 07's explicit design comment that this subtraction should NOT happen (a weir is a real modelled barrier; subtracting a scalar from the boundary assumes a uniform wall with no SFINCS-modelled barrier at all) -- also double-counts protection, since the same FLOPROS number becomes both the boundary offset and the weir crest floor. Unrelated to the crest bug above; needs its own decision.

# 2026-08-02b: removed rule 13b (validate_protection_level) entirely (- JS)

Follow-up to the scenario-rerun investigation below, which flagged that `13b_validate_protection_level.smk` hard-coded `scenarios/default/...` in its own inputs (even though its own outputs are basin-level), pulling an extra full `build_sfincs` + simulation for the `default` scenario into the DAG on every `build` request, regardless of which scenario was actually being built. Rather than scope the rule to whichever scenario is actually requested, the decision was to remove the rule entirely.

Removed: `workflow/rules/13b_validate_protection_level.smk`, `workflow/scripts/13b_validate_protection_level.py`, the `include:` line and `_BUILD_OUTPUTS` block in `workflow/Snakefile`, `sfincs.protection_validation.*` in `config/config.yml`, and `src.river_depth_calibration.check_convergence()` (a windowed-flatness convergence check written specifically for this rule -- it had no other caller; rule `modelled_depth_estimation`'s own calibration deliberately never used a windowed convergence check, it always used the period-maximum water level instead, see that rule's own module docstring). The basin-level `visuals/model_runs/` directory no longer exists at all, since this rule was its only user.

Verified via `snakemake -n`: job count for a single-scenario build dropped from 14 to 12 (removing both the validation job itself and the extra `default`-scenario `build_sfincs` job it was forcing).

# 2026-08-02: scenario switch was re-triggering the entire preprocessing chain, including depth calibration (- JS)

User reported that running a different scenario (`target_scenarios=['coast_100']`) after other scenarios had already been built re-ran the WHOLE preprocessing pipeline for the basin, including the expensive SFINCS-based depth/crest calibration (rule `modelled_depth_estimation`) -- even though no input data had changed, only the requested scenario. Investigated via `snakemake -n` dry runs comparing job stats/reasons across different `target_scenarios` values.

**Root cause**: `workflow/rules/07_boundary_forcings.smk` (rule `get_boundary_forcings`, a BASIN-level rule with no `{scenario}` wildcard -- it runs once per basin, before the scenario axis branches) had a `params:` entry `design_rp_river_yr = SCENARIO_DEFS[SCENARIOS[0]]["river_rp"]` -- i.e. it read *the first entry of whatever `target_scenarios` list was passed on the command line*. Snakemake's default rerun-triggers include `params`, so this single line meant rule 07's own recorded metadata (and therefore its own "needs rerun?" status) depended on which scenario(s) were requested and even their LIST ORDER -- confirmed empirically: `target_scenarios=['default','coast_100']` did not rerun rule 07, but the reversed `target_scenarios=['coast_100','default']` did, with identical code and data. Once rule 07 is marked dirty, everything downstream cascades: 08 -> 08b -> 08c -> 09 -> **10 (`modelled_depth_estimation`)** -> 13 and on.

The value itself was only ever used for a *diagnostic preview* inside `07_get_boundary_forcings.py` (a "how big is this crossing" plot annotation and the EVA return-value used for the `has_glofas`/`ok` gate) -- the actual production `discharge_rp_table` (consumed by rules 08/10/13) is computed at every standard return period regardless of this value, and rule 13 builds each scenario's own REAL forcing from its own actual `design_rp_river_yr` (via `scenario_params()`), never from this diagnostic value.

**Fixed**: this diagnostic RP is now a fixed, scenario-independent config value (`boundary_forcings.river.eva.rp_fl`, default 100.0) instead of being derived from the requested scenario list. `design_rp_river_yr` param removed entirely from rule 07; `07_get_boundary_forcings.py` now just reads `rp_fl` straight from `params.eva` (already scenario-independent) instead of overriding it. Verified via dry run: rule 07's own recorded "now" params are now byte-identical across `coast_100`, `river_100`, and reversed scenario-list orderings. One transitional rerun of the whole preprocessing chain is still unavoidable the next time ANY scenario is built (the on-disk metadata reflects the old, scenario-dependent param structure) -- but every subsequent scenario switch after that will no longer touch preprocessing.

**Side benefit**: this also fixes a latent correctness bug, not just a performance one -- `has_glofas`/EVA "ok" gating (which crossings get modelled at all) previously could silently differ for the SAME basin depending on which scenario happened to be requested first, since it was gated on `isfinite(q_rp100)` computed at the scenario-derived RP. It's now a stable, basin-level computation.

## Also fixed while investigating: latent crash building `coast_100`/`river_100` scenarios

`13_build_sfincs.py` unconditionally did `design_rp_river_yr = float(snakemake.params.design_rp_river_yr)` and the same for `design_rp_surge_yr`. For scenarios with a `null` RP on one side (`coast_100`: `river_rp=null` -> `forcing_mode="coastal_only"`; `river_100`: `surge_rp=null` -> `forcing_mode="river_only"`), this is `float(None)`, which raises `TypeError` -- would have crashed the very first real (non-dry-run) build of either scenario. Not caught by `snakemake -n` since it's a runtime error inside the script, not a DAG-building one. Fixed: both params now stay `None` when the underlying scenario value is `None`, matching what their consumers already expect -- `src.river_forcing.build_design_discharge_matrix` already accepts `design_rp_yr: float | None` natively (falls back to a constant bankfull hydrograph), and `design_rp_surge_yr` is only ever dereferenced inside the `forcing_mode != "river_only"` branch, i.e. exactly when it's guaranteed non-None.

## Flagged, not fixed -- needs a decision

`13b_validate_protection_level.smk` hard-codes `scenarios/default/...` in its own inputs even though its outputs are basin-level. This pulls an EXTRA full `build_sfincs` + simulation for the `default` scenario into the DAG on every `build` request, regardless of which scenario was actually asked for (visible in dry-run job stats: `build_sfincs 2` even when only requesting `coast_100`). May be an intentional design choice (validate protection level against one canonical reference scenario rather than per-event) -- left untouched pending a decision on whether it should be scoped differently.

# 2026-07-29d: per-round diagnostics genuinely skipped instead of empty-touched, filenames renamed for cross-round comparison (- JS)

Two refinements to the per-round diagnostic plots added earlier the same day:

- **Genuinely skipped, not empty-touched.** Rounds skipped by early-stopping used to get an empty (0-byte) placeholder file for each of the 5 visual outputs, since they were declared Snakemake outputs and Snakemake requires every declared output to exist. They're no longer declared Snakemake outputs at all (written directly under `calib_root`, same convention `calibration_state.csv` already used) -- a skipped round now gets NO file whatsoever. Only the FINAL round slot (`round{n_correction_iterations}`) still gets a copy of the accepted round's own real diagnostics, since the canonical (non-indexed) outputs are copied from exactly that slot.
- **Round number moved to the end of the filename.** `round4_max_inundation.png` -> `max_inundation_round4.png` (and the same for the other 4 kinds). A directory listing sorted alphabetically now groups by PLOT KIND first, so comparing the same diagnostic across rounds (e.g. every `max_inundation_round*.png`) means looking at consecutive files instead of picking them out of a listing interleaved by round.

# 2026-07-29c: calibration discharge now ramps from bankfull instead of stepping instantly to the full value (- JS)

Investigating why a live run on basin 2433835 showed wild, physically-implausible water-level oscillation (multi-metre swings hour-to-hour, under CONSTANT discharge forcing) right at the seed reach's own discharge-injection cell: `10_depth_estimation_modelled.py`'s own discharge forcing was a flat step function -- the full calibration discharge (here, 3276 m³/s, a 50.8-year return period peak, vs. 974.8 m³/s bankfull) applied from t=0 with no ramp at all, into a channel that starts near-dry. Production's own forcing (`src.river_forcing.build_design_discharge_matrix`/`sinusoidal_wave`, `14_run_spinup.py`) never does this -- it always starts at a bankfull baseline for a lead period before ramping to the event peak, specifically to avoid this kind of startup shock.

Fixed: the calibration discharge now ramps LINEARLY from each crossing's own `bankfull_discharge` up to the full calibration discharge over a new `river_processing.river_depth_modelling.discharge_ramp_hours` (default 6.0), then holds flat for the remainder of the run -- simpler than production's sinusoidal hydrograph since calibration only needs to reach and hold the peak, not simulate a full event recession. A crossing with a missing/NaN bankfull value (shouldn't happen, but defensively) ramps from 0 instead of skipping the ramp. Verified with a standalone test of the ramp math (starts exactly at bankfull, reaches the target discharge exactly at `discharge_ramp_hours`, holds flat afterward, NaN-bankfull fallback works).

Not yet re-verified against a live SFINCS run at time of writing -- next step is re-running basin 2433835's calibration to confirm the seed-cell oscillation is actually resolved.

# 2026-07-29b: small dikerings pinched onto the main boundary removed in vector space (- JS)

Found inspecting the Mississippi basin's traced weir gpkg: `discard_small_components` (raster level) only ever drops a feature that is its OWN separate connected component. A small feature attached to a larger, kept component at a single diagonal-pinch pixel — exactly the geometry the earlier 8-connectivity fix correctly keeps as "part of the larger component" from a masking standpoint — still traces its own tiny closed ring in the final VECTOR output, meeting the main boundary at one shared 4-way vertex (two segments continuing the main boundary through that point, two entering/leaving the small ring). `discard_small_components` never sees this case, since at the raster level the small feature is already merged into one large component.

New `protection_weir._remove_small_dikerings()`, called at the end of `build_coastal_protection_weir()` right after `weir_gdf` is assembled: builds a vertex→incident-segment graph from `weir_gdf`'s own segment endpoints, and at every vertex where exactly four segments meet, traces each incident direction through a chain of ordinary degree-2 vertices to see if it closes back into a small loop at that same vertex. If the loop's own enclosed area (in grid cells, via the loop's polygon area divided by `cell_size_m²`) is under `min_component_cells`, every segment forming that loop is dropped — the two "through" segments at the pinch vertex are always kept, so the main boundary passes straight through with no gap, exactly mirroring `discard_small_components`'s own "no ring around a too-small feature" outcome, just reached in vector space. A generous trace-length cap avoids walking the entire remaining network for every pinch point on a coastline with tens of thousands of segments. Verified with a synthetic test: a small (0.5-cell) diamond ring pinched onto a 20-segment main chain is fully dropped with the main chain left intact, while a large (100-cell) ring at an identical pinch topology is correctly left untouched.

# 2026-07-29: calibration crest redesign -- cross-section zs, direct per-cell target, no interpolation (- JS)

Six related changes to `10_depth_estimation_modelled.py`'s correction-round crest logic, found while investigating why basin 2433835's round-6 snap verification failed (round 5's snap tightened the crest to `zs + freeboard_m`, but the SMOOTHED (interpolated) painted crest ended up below the actual simulated water level at 2 cells once run).

## New: per-centerline-cell zs sampled from its own channel CROSS-SECTION, not the single intersected cell

A single SFINCS grid cell's own water level can be a noisy, non-representative sample of "the water level at this point along the river" -- the real cross-section at that point is usually several cells wide, and the centerline only intersects one of them. Every centerline cell's own period-max water level (driving every depth/crest calibration decision) is now the MAX across the full cross-section: the UNION of the contiguous `channel_mask` run through the cell along its own grid ROW AND along its own grid COLUMN (each extended by one cell to also reach the adjacent land cell the weir itself sits on). New `_read_cross_section_period_max_zs`; `_read_zs_and_resolve` (single intersected cell) is kept unchanged, now used ONLY for the water-level-timeseries diagnostic plot's own representative-cell indexing, a visualization concern separate from calibration.

**Investigated same day**, after a live run on basin 2433835 exposed a real consequence: round 4 converged and the one-shot snap tightened the crest, but round 5's verification overtopped 2 cells by -0.144 m. Root-cause investigation (reusing round 4/5's own already-existing `sfincs_map.nc`/`calibration_state.csv`, no new SFINCS run) found the column-run (4 cells, chosen since shorter than the 7-cell row-run) pulled in a cell whose own reading (6.985 m) was substantially higher than the 2 cells the crest was actually being verified against (6.53/6.55 m). A union-of-both-directions variant was tried and reverted -- confirmed degenerate for any reach running straight along one grid axis (the OTHER axis's own run is then the reach's entire length, giving every cell on that whole reach the same single global maximum). Further investigation (real-world along-reach projection, natural and burned DEM, `channel_mask`'s own footprint) confirmed the pulled-in cell is genuinely part of the same excavated, connected corridor -- just a fringe cell included via `all_touched=True` rasterization slightly beyond the reach's own declared half-width, not a selection bug. Kept the shorter-run-wins mechanism (with the one-cell land-side extension added this session); the round-5 snap failure stands as a real, valid finding that the existing snap-verify-revert safety net correctly caught, not something requiring a further code fix.

## Changed: correction-round crest update is now a direct per-cell target, no smoothing, no two-case branching

Replaced the old `badly_overtopped` (close the full gap) / `near_zero` (flat increment) two-case update with a single unconditional formula applied to every cell: `weir_crest_current = period_max_zs + min_crest_increment_per_round_m`. This can raise OR lower a cell's crest relative to its current value (previously crest was raise-only until the one-shot snap). Once a round's own crest already exceeds that round's own period-max water level everywhere, the existing one-shot snap-and-verify step (unchanged: tighten to `zs + freeboard_m`, run one more round, revert on failure) still applies.

## New: `build_nearest_weir_crest_regular` -- no-interpolation weir painting

The correction-round crest is now painted onto the grid with a NEW function (`river_burn.build_nearest_weir_crest_regular`) that assigns every raster cell its NEAREST centerline anchor's crest value directly (cKDTree nearest-neighbour, any reach), instead of `build_smoothed_weir_crest_regular`'s along-reach interpolation + junction blending. Since the per-cell targets are now already exact requirements, interpolating between them reintroduced the same under/over-shoot smoothing was meant to avoid -- this was the direct cause of round 6's snap failure on basin 2433835. `build_smoothed_weir_crest_regular` itself is untouched and still used by rule 13's quadtree fallback; rule `modelled_depth_estimation` no longer passes it `weir_crest_junction_blend_m` at all (removed from that rule's own `.smk` params -- still used by rule 13's own quadtree fallback param passing, unaffected).

## Fix: diagnostic backfill was overwriting a round that actually ran

When the one-shot snap FAILED, `converged_round_idx = round_idx - 1`, but the backfill loop's range (`converged_round_idx+1 .. n_correction_iterations`) included `round_idx` itself -- the just-simulated (rejected) verification round -- overwriting its own genuine `calibration_state.csv`/plots/animation with copies of the earlier accepted round's files. Confirmed via byte-identical MD5 hashes of what should have been two different rounds' `max_inundation.png`. Fixed: rounds that never actually ran get an honest empty placeholder instead of a misleading copy (their `calibration_state.csv` still gets a harmless copy, since no genuine data ever existed there to misrepresent, and `gather_calibration_round_profile` reads every round index unconditionally); only the FINAL slot (`round{n_correction_iterations}`) gets a full copy of the accepted round's own files, since the canonical-output copy step reads from exactly that slot. (Originally an empty `.touch()`'d file for the 4 visual outputs -- superseded the same day by 2026-07-29d below, which un-declares these as Snakemake outputs entirely so a skipped round gets no file at all.)

## Fix: exported weir gpkg could reflect a rejected (failed-snap) crest

On a failed snap, `weir_crest_current`/`coastal_crest_current`/`river_boundary_crest_current` were reverted to their pre-snap values, but the `weir_gdf` object actually written to `coastal_protection_weir.gpkg` was never rebuilt -- it stayed whatever the FAILED, over-tightened snap round had traced. Fixed: on a reverted snap, `weir_gdf` is re-traced once, after the loop, from the now-reverted (accepted) crest arrays via `build_nearest_weir_crest_regular` + `build_coastal_protection_weir`, before being written.

## Fix: diagnostic plot showed the seed reach's own line-start, not the real discharge point

The (now-superseded) ad hoc diagnostic script plotted `seed_reaches.geometry.coords[0]` as "the discharge point" -- the seed reach's own raw line-start vertex, not where discharge is actually injected, and it visibly sat outside the embankments. New per-round diagnostic (`plots.plot_crest_gap_map`, wired into the main calibration loop as a 5th per-round output, `crest_gap_map_round{i}.png` / canonical `crest_gap_map.png`) instead plots `crossings_utm` -- the same snapped discharge locations `sf.discharge_points.create()` is actually forced with. The plot also auto-detects imshow orientation from the grid transform's own sign convention (a SFINCS grid commonly has a positive y-scale, which a plain `imshow()` silently flips), and overlays the weir line + river network on both panels.

# 2026-07-28: four calibration/build fixes found investigating basin 4267691/2433835 (- JS)

Four independent, previously-validated fixes, reimplemented after an unrelated same-day redesign attempt (per-segment weir-crest tracking) was reverted for producing a regression; these four are unaffected by that revert and stand on their own.

## Fix: `compute_max_inundation`/`compute_flood_timeseries_stats` skipped memory bounding for regular grids

`_coarsen_for_memory()` — the safety net that bounds the subgrid reference raster (`get_bed_level()`'s `dep_subgrid.tif`) to a sane size before use — was only ever invoked `if isinstance(da_zsmax, xu.UgridDataArray)`, i.e. quadtree grids only. Rule `modelled_depth_estimation` always builds a REGULAR grid (it has its own `NotImplementedError` guard against quadtree), so this condition was always false there, and the full-extent subgrid raster (basin 4267691: ~700M pixels / ~2.8 GB uncompressed) was used completely unbounded, crashing round 0's own diagnostics with no traceback (memory exhaustion). Fixed in both `compute_max_inundation()` and `compute_flood_timeseries_stats()` by dropping the quadtree-only gate — `_coarsen_for_memory()` already has its own internal size check and is a no-op below `max_bytes`, so calling it unconditionally is safe for both grid types and was never actually quadtree-specific. Affects rule 16's flood-timeseries stats too for any sufficiently large regular-grid basin.

## New: discharge points snapped onto the network centerline, not just the active region

Neither hydromt_sfincs's own `discharge_points.create()` nor this codebase's existing point-wrangling (`geometry.snap_points_into_region`, which only nudges a point back inside the model's active region) ever checked that a discharge point's resolved SFINCS grid cell actually sits on the modelled channel. A crossing's domain-entry point is geometrically exact (computed directly from the reach's own line geometry in rule 07), but nothing guaranteed the grid cell it rasterizes onto was part of `channel_mask` — on a coarse enough grid it could land on a neighbouring floodplain cell with no real conveyance instead. This is very likely what produced basin 4267691's round-0 calibration water-level spike (369 m at one location): a large constant discharge injected into a cell with no channel depth to carry it.

- New `river_burn.snap_points_to_centerline_cells()`: snaps each point onto the nearest cell that `build_centerline_cells_regular()` says the reach's own RAW centerline actually intersects (not just the buffered `channel_mask` corridor). When the point's own `inside_reach_id` is known (river_forcing.nc), the search is restricted to that reach's own cells first — avoids snapping to a *different* nearby reach's cell at a confluence/bifurcation — falling back to a search across all reaches otherwise. Logs a warning if any point snaps implausibly far (>3 grid cells), which usually means `inside_reach_id` resolution failed rather than a genuinely distant channel.
- Wired into both `10_depth_estimation_modelled.py` (reuses its own already-computed `cell_gdf`, zero extra cost) and `13_build_sfincs.py`'s regular-grid discharge-forcing path, right before the existing region-boundary snap.
- **Known gap, scoped out for now:** regular grid only. There is no quadtree equivalent of `build_centerline_cells_regular` yet (mesh faces vs. a simple affine raster transform is a different algorithm), so quadtree production builds keep the previous unsnapped behaviour until that's written.

## New: seed reaches get a flush channel-buffer cap, not a rounded bulge

Every per-reach channel buffer in `src/river_burn.py` (`channel_mask`, the burn excavation corridor, the weir-smoothing corridor) is built via a plain `line.buffer(width / 2.0)`, which defaults to a ROUND end cap at both ends of every reach. That's harmless at an internal junction (a neighbouring reach's own buffer already overlaps and covers the join regardless of angle), but wrong at a network SEED reach (`is_seed=True`, no upstream neighbour): the round cap bulges out in an arc beyond the line's own true start vertex, enclosing extra channel cells that `build_centerline_cells_regular` never samples (it only follows the raw line, not the buffer) — invisible to per-cell calibration tracking, yet inside the same confined pocket as the actual discharge-injection cell, and potentially feeling worse pileup given the tapering rounded shape.

- New `river_burn._flush_capped_buffer()` (+ `_half_plane_beyond()` helper): same `line.buffer(width/2)`, but clips the round cap at the requested end(s) back to a flat cut exactly at that endpoint, perpendicular to the line's own local tangent there. Degenerate first/last segments fall back to the unclipped round cap rather than raising.
- Wired into all four existing `line.buffer(...)` call sites (`burn_river_channel`, `_channel_buffer_polygons` → `build_channel_mask_regular`, `build_smoothed_weir_crest_regular`, `build_smoothed_weir_crest_quadtree`) with `clip_start=bool(row.is_seed)` — every other reach keeps its plain round-capped buffer unchanged.
- **Scoped out for now:** terminal/mouth reaches (no downstream neighbour) have the same rounded-cap bulge, but a mouth's downstream end transitions into the coastal weir/`ocean_mask` handling through a different mechanism than "another reach's buffer covers the round cap" — flush-capping there isn't confirmed correct yet, so `clip_end` is left `False` everywhere pending its own verification.

## New: weir topology guaranteed continuous — no more dangling gaps

Found by directly inspecting `2433835_coastal_protection_weir.gpkg`: connectivity analysis showed 38 dangling endpoints out of 3547 segments — real breaks in the traced coastline/riverbank, not just a rendering artifact. Requirement: every weir segment's endpoint must either connect to another segment, or lie on the delta polygon's own boundary — the result must resolve into continuous boundary-to-boundary chains or fully closed loops, never a gap in the middle.

Root cause, in `protection_weir.py`'s edge-tracing functions (`_seaward_edges_regular_with_values` / `_seaward_edges_quadtree_with_values`, shared by both regular and quadtree grids via `seaward_edges_with_values`): a land/water_like transition is detected purely geometrically, but the segment is then silently DROPPED if that cell's crest value (`crest_surface`) is `NaN`. `crest_surface` is `max(elsewhere_crest, river_crest_on_grid)`, and while `river_crest_on_grid` legitimately has NaN gaps beyond its own dilation radius, `elsewhere_crest` (`coastal_crest_on_grid`, rule modelled_depth_estimation's own per-cell probe-derived array) can ALSO have NaN holes beyond its own probe/dilation coverage — e.g. a land cell next to a small isolated water patch that's neither a modelled river reach nor near a coastal probe. Wherever both are NaN at once, the segment silently vanished, breaking an otherwise continuous boundary.

Fixed in `build_coastal_protection_weir()`: `elsewhere_crest` now falls back to the flat `crest_elevation_m` scalar wherever `coastal_crest_on_grid` itself is NaN — exactly what "no per-cell override available" already means everywhere else that array doesn't cover. This guarantees `crest_surface` is finite at every land cell, so the edge tracer's NaN-drop branch never actually triggers for this call path — every geometrically-real land/water_like transition now gets a segment, unconditionally. Verified with a synthetic unit test (a straight 10-cell coastline with a deliberate NaN hole at one cell — now produces exactly 10 segments, none dropped, with the NaN cell correctly falling back to the flat scalar crest).

## Fix: weir topology gaps at a diagonal-only coastline pinch (4-connectivity artifact in `discard_small_components`)

Testing the four fixes above by tracing basin 2433835's weir geometry directly (no SFINCS run needed — weir line positioning depends only on the grid/`channel_mask`/land classification, not on any calibrated crest) still showed 4 dangling endpoints, even with the NaN-drop fix above already in place. Investigation (dumping `land_mask`/`water_like` around each dangling point) found 2 of the 4 traced back to a second, distinct bug: `GridArrays.connected_components()` (the only caller of `discard_small_components`, which drops land/water patches smaller than `min_component_cells`) used `scipy.ndimage.label()`'s default 4-connectivity (only edge-sharing neighbours count as connected). A real, physically continuous coastline commonly narrows to a single DIAGONAL-only pixel-to-pixel connection at some point (an ordinary raster-discretization artifact for a jagged coastline, not a genuine break) — under 4-connectivity that diagonal touch doesn't count as "connected," so a large mainland and a small diagonally-attached tail get classified as two separate components, and the small tail (confirmed: ~7 cells, well under `min_component_cells: 30`) gets dropped as if it were an isolated island. The dropped cells become neither `land_mask` nor `water_like`, and the edge tracer draws no segment against either side of them — the boundary just stops dead at the pinch point instead of continuing along what is, physically, the same landform.

Fixed by switching `connected_components()`'s regular-grid branch to 8-connectivity (`generate_binary_structure(2, 2)`, treating a diagonal touch as connected) — confirmed via a synthetic unit test (a 25-cell block + a 4-cell tail joined only diagonally: previously the tail was dropped, now both are kept as one 29-cell component) and via re-tracing basin 2433835's actual weir (4 dangling endpoints → 2; the remaining 2 are a distinct, minor issue in far corners of the rectangular grid bounding box with no land or channel nearby at all, not this bug).

## Fix: mouth crest was the raw seabed elevation, never verified against the real coupled water level there

Requested after reviewing how `10_depth_estimation_modelled.py`'s final crest (`zs + freeboard_m <= crest`) holds up along the whole river: it works well wherever the centerline actually tracks a cell, but the mouth reach's own last cell was a special case that bypassed verification entirely. Every round (round 0, every correction round, and even the final exact-tightening snap) unconditionally set `weir_crest_current[is_last_mouth_cell] = dem_at_cell[is_last_mouth_cell]` — the raw seabed elevation, only ever floored afterward by the flat `coastal_protection_crest_m` scalar. Unlike literally every other tracked cell (and even the coastal/river-boundary *probe* cells), the mouth's own crest was never checked against its own round's actual simulated water level at all.

Fixed: the mouth's own crest is now `max(period-max water level at the nearest OCEAN-classified cell + freeboard_m, coastal_protection_crest_m)` (`_mouth_crest_target`, sampled via a new small "mouth ocean probe" — one nearest-ocean-cell per mouth reach, same `_resolve_map_cell_idx`/`_read_map_zs_at_cells` machinery every other probe already uses). Sampling the OCEAN side specifically (not the mouth's own river-side channel cell) was a deliberate choice — that channel cell sits right at the coast/river transition and its own reading isn't necessarily representative of the open water the crest there actually has to hold back, mirroring the existing river-boundary probe's own "look across the dike" logic. Bed/depth at the mouth is unchanged (still hard-forced to natural bathymetry, a real physical constraint, not a calibration target) — only the crest formula changed, at all three places it used to be hard-forced.

## New: discharge-buffer land cells sample the worst (max) water level in their own radius, not just the nearest channel cell

Related follow-up: near a seed/head reach, the discharge injection cell's own water level is not necessarily the local peak (a narrow/bottlenecked head can push the true peak a cell or two away — see this module's own long-standing "sharp local spike" caveat about the injection cell). The existing river-boundary probe mechanism sampled every probe cell (including ones inside a discharge point's own 5-cell buffer) at a single nearest-channel-cell reading, which could miss that off-injection-cell peak.

Fixed: `river_boundary_probe` cells that fall inside a discharge point's own buffer zone (`_discharge_buffer_land`) now sample the MAX water level across every channel cell within that same radius of their nearest discharge point, instead of one nearest-cell reading (falls back to the nearest-cell reading if that local neighbourhood is somehow empty). Ordinary dike-adjacent probes elsewhere (not near any discharge point) are unaffected. Since the existing weir-build combines the centerline's own per-cell crest and this probe correction via `np.maximum`, the centerline's own (more precise, round-verified) value still wins wherever it's higher — this change only ever raises protection further where the centerline is silent or lower, never suppresses a real local peak with a too-low single-point reading. Verified with a synthetic test of the new index/aggregation logic (ordinary probe reads its own nearest cell; discharge probes both correctly pick up an off-injection-cell peak; empty-neighbourhood fallback works).

---

# 2026-07-29: weir topology gaps from discarded small islands/patches reclassified instead of left as holes (- JS)

Testing the weir generation process standalone for basin 4267691 (much bigger/more complex coastline than 2433835) via the same SFINCS-free geometry-only trace used earlier found 22 dangling endpoints. 6 of them traced back to a THIRD distinct weir-topology-gap mechanism, on top of the NaN-drop fix and the diagonal-4-connectivity fix already shipped: `discard_small_components()` (called on both `land_mask` and `water_like` separately, to drop small islands/patches below `min_component_cells`) leaves every discarded cell as neither `land_mask` nor `water_like` — invisible to the edge tracer (`h_break`/`v_break` require one side True and the other True; a cell reading False on both gets no segment drawn against it from either direction). Where a discarded small feature happens to sit at the true edge of a larger, still-kept feature, this doesn't just skip a ring around the excluded feature itself (the intended behaviour) — it also breaks the LARGER feature's own boundary trace at that exact point, leaving a dangling gap in what should still be a continuous coastline.

Fixed in `build_coastal_protection_weir()`: a cell discarded from `land_mask` is now reclassified into `water_like` instead of being left as a hole, and vice versa for a discarded `water_like` cell reclassified into `land_mask` — a too-small island dissolves into its surrounding water, a too-small pond dissolves into its surrounding land, either way producing a genuine, continuous transition at the larger feature's own true edge (or no segment at all, where the discarded feature was fully isolated), achieving the original "no ring around a small excluded feature" intent without leaving an untraceable gap behind. `land_mask_raw`/`water_like_raw` are mutually exclusive by construction, so the two reclassified sets can never collide with each other or with the surviving mask on the opposite side (proven via a synthetic exclusivity/coverage test, not just asserted).

Verified empirically on both basins: 2433835 unaffected (4010 segments now vs. 3988 before — the extra segments are newly-traced boundaries around dissolved small features — still exactly the same 2 pre-existing, distinct array-edge dangling points, no regression). 4267691: 22 → 16 dangling endpoints — every single one of the 6 resolved was a genuine coastline small-feature case; the remaining 16 are now *exclusively* the array-bounding-box-edge case (all sitting at the grid's own exact left/right raster edge coordinates, 14–36 km from any river or channel) — a separate, lower-priority, not-yet-investigated mechanism, unrelated to this fix.

---

# 2026-07-26: merge `main`, consolidate depth-estimation rules, and get modelled river-depth calibration fully working (- JS)

Everything below covers `improve_base_model` since the "improve_base_model
(merged as PR #1)" entry further down — a large amount of work (some of it
already committed in "Renumber depth-estimation/testing rules, restructure
config, and clean up narrative comments", the rest still uncommitted at the
time of writing) that hasn't been written up until now. Several items below
supersede parts of the PR #1 entry (most notably "River DEM burning (new
feature)", which described a `burn_river_dem` rule that no longer exists in
this form — see "Depth-estimation rules consolidated" below).

Merged `origin/main` (PR #2, "scenario assessment + flood metrics table",
see the 2026-07-20 entry below for what it introduced) into
`improve_base_model`. Both branches had evolved independently since PR #1 —
`improve_base_model` had renumbered/restructured several rules and config
sections in the meantime (see the immediately-preceding commit,
"Renumber depth-estimation/testing rules, restructure config, and clean up
narrative comments") — so this was a real merge with real conflicts, not a
fast-forward.

## Conflict resolution highlights

- `sfincs.simulation.sfincs_exe` (`config.yml`): git auto-merged this to the
  colleague's own personal machine path with **no conflict marker** at all
  (only one side had touched that specific line) — reverted to the local
  path. A machine-specific value like this should go through the
  `GCFM_SFINCS_EXE` env-var override (`00_common.smk`) rather than being
  hand-edited in a shared, tracked file; worth double-checking after any
  future merge since git will not flag this kind of collision.
- Restored `postprocessing.WATER_LANDUSE_CODES = (0, 2000)`. Main's PR
  "corrected" this to `(80, 200)` (see the 2026-07-20 entry's own Fixes
  section) but that reverts a deliberate earlier fix: `(0, 2000)`
  intentionally matches no real land-use class so rivers/ocean stay
  unmasked in these diagnostics — confirmed intentional in this session's
  own project notes, not the bug the "fix" assumed it was.
- Rule `build_sfincs`'s output list (main's scenario refactor) had dropped
  `sfincs_weir`/`weir_gpkg`/`plot_coastal_protection_weir` from `output:`,
  and the `river_only_flat_level_m` param entirely, from both
  `13_build_sfincs.smk` and the `Snakefile`'s own `_BUILD_OUTPUTS` list —
  all three still actively used by `13_build_sfincs.py`'s script body (the
  weir outputs would have stopped being tracked by Snakemake; the missing
  param would have raised `AttributeError` the moment `forcing_mode ==
  "river_only"`). Restored, moved under `scenarios/{scenario}/` like
  everything else in that rule.
- `13b_validate_protection_level.smk` (untouched by main's own PR) still
  pointed at the pre-scenario flat `{basin_id}/sfincs/` path, now dead
  since rule 13 writes under `{basin_id}/scenarios/{scenario}/sfincs/`.
  Repointed at the `default` scenario's own build specifically — protection
  validation is about the standard/default configuration, not an
  exploratory scenario. Confirmed safe to do: grid/elevation/mask/weir/
  roughness/subgrid are byte-identical across every scenario for a given
  basin (only `sfincs.inp`'s own timing and the `sfincs.dis`/`sfincs.bzs`
  forcing files vary) — `design_rp_river_yr`/`design_rp_surge_yr` are
  consumed strictly downstream of weir/grid/mask construction in
  `13_build_sfincs.py` (sections 9–10, after the weir is built in 4c).
- `river_forcing.build_design_discharge_matrix`: merged main's new
  `design_rp_yr=None` → constant-bankfull ("mean conditions") branch with
  the pre-existing `apply_protection_floor` gating, which main's own
  version had silently dropped — keeping the floor disabled under
  `depth_method="modelled"` avoids double-counting protection already
  represented by the calibrated weir. Also replaced main's inline
  duplicate of the log-RP interpolation with the existing
  `interpolate_discharge_at_rp` helper rather than carrying two copies of
  the same lookup forward.

## Forcing-mode derivation redesign

Main's PR left `mode` a step short of actually being scenario-driven: named
scenarios always hardcoded `"compound"` regardless of their own RPs (a
`null` RP fed a "mean conditions" hydrograph/tide rather than the
`river_only`/`coastal_only` *isolation* its absence was presumably meant to
signal), and the `default` scenario's own RPs still lived in `config.yml`'s
old `boundary_setup` keys rather than the new scenario mechanism.

- New `derive_forcing_mode(river_rp, surge_rp)` (`src/river_forcing.py`):
  both RPs set → `"compound"`; exactly one set → `"river_only"`/
  `"coastal_only"`; neither → `ValueError` (a scenario needs at least one
  real driver to build a model at all). Shared by `00_common.smk`'s
  `scenario_params()` and any standalone script that needs to replicate a
  scenario's own mode outside Snakemake, so the derivation logic exists in
  exactly one place.
- Removed `sfincs.boundary_setup.mode` and
  `sfincs.boundary_setup.design_rp_river_yr` from `config.yml` entirely —
  mode is always derived now, never configured directly. `default` is now
  a real entry in `config/scenarios.yml` (`river_rp: 150`, `surge_rp: 100`
  — the old values, preserved) instead of a hardcoded special case reading
  `config.yml`'s old `boundary_setup` keys.
- Removed the `baseline` scenario (`surge_rp: null, river_rp: null`) from
  `scenarios.yml` — under the new derivation rule this is invalid (no
  driver at all) rather than a usable "mean conditions everywhere" case.
- New `testing.upstream_boundary_check.surge_rp: 100` (must be a COAST-RP
  tabulated value) — rule 11's own wave-propagation sanity check now looks
  this RP up directly from `surge_forcing.nc`'s full `storm_tide_rp_table`
  via new `surge.lookup_storm_tide_at_rp()` (factored out of the existing
  `build_design_surge_matrix`), decoupled from both rule 07's own
  station-selection RP and any scenario's own design RP.
- Rule 07's diagnostic discharge-preview plot and three test scripts
  (`test_discharge_sensitivity.py`, `test_discharge_return_period_
  response.py`, `test_grid_resolution_benchmark.py`) now resolve
  `config.get("target_scenarios", ["default"])[0]` for their own RP/mode
  references (mirroring `00_common.smk`'s own `SCENARIOS` resolution),
  instead of hardcoding `"default"` regardless of what's actually targeted.
  `test_discharge_return_period_response.py` also had its own,
  independent break from the scenario path move (still reading production
  build files from the old flat `results/{basin_id}/sfincs/` path) fixed
  at the same time.

## Also removed in this pass

`tests/test_grid_resolution_benchmark.py` and its exclusively-used helper
`tests/_snakemake_script_runner.py` (confirmed via repo-wide search: no
other consumer), plus their output directories
(`figs/grid_resolution_benchmark/` and
`D:/GCFM_UU/experiments/grid_resolution_benchmark/`, ~7.5 GB). The
benchmark's own hand-mocked `snakemake.params`/`snakemake.output` for
`13_build_sfincs.py` had drifted out of sync with that script's evolving
interface (missing the `active_mask_enabled`/`active_mask_elevation_
buffer_m` params and `river_elevation_max` input added earlier this
branch, on top of the scenario-mechanism params above) and was no longer
being kept current.

## Depth-estimation rules consolidated: `empirical`/`modelled` as sibling rules, everything else made unconditional

The old `09a_river_depth → 09b_estuarine_depth → 10_condition_elevation →
11_river_preburn → 11b_burn_river_dem → 12_testing` five-rule chain (all
individually opt-in via `river_processing.conditioning.enabled` /
`burn_rivers.enabled`) is gone. Current shape:

- **`condition_elevation` (rule 09, was rule "10") now ALWAYS runs** —
  `river_processing.conditioning.enabled` was removed entirely, there is no
  toggle anymore. It only needs rule 08's cleaned network topology (not
  discharge/depth), so it now runs *before* depth estimation instead of
  after, and additionally writes a second output resampled onto a new
  shared SFINCS grid (see "New rules `08b`/`08c`" below) so every later
  consumer of the conditioned elevation uses the byte-identical raster.
- **New `river_processing.depth_method: "empirical" | "modelled"`** selects
  which of two SIBLING rules is *defined at all* — `empirical_depth_
  estimation` or `modelled_depth_estimation` (both numbered rule 10) —
  guarded by a module-level `if` in each `.smk` file (not a runtime
  branch), since both would otherwise declare the same output filenames and
  Snakemake would raise `AmbiguousRuleException`. Every downstream
  consumer (rule 11, rule 13) reads the same unified output filenames
  regardless of which one ran.
- **River-bed burning is now ALWAYS produced by whichever rule-10 sibling
  ran** — `river_processing.burn_rivers.enabled` was removed entirely.
  There is no more standalone `zbed_anchors.gpkg` file: each sibling calls
  `compute_river_bed_points()` (`src/river_preburn.py`) immediately followed
  by `burn_river_channel()` (`src/river_burn.py`) in the same script,
  writing `river_burned_dem.tif` (native resolution, for subgrid) AND a new
  `river_burned_dem_sfincs_grid.tif` (SFINCS-grid resolution, for the main
  "dep" layer) directly — no separate preburn/burn-DEM rule, and no more
  `hydromt_sfincs` `gdf_zb`/`burn_river_rect` path in rule 13 at all.
- The empirical sibling keeps the Leopold-Maddock power-law depth (now
  `depth = c · Q^f` directly against `bankfull_discharge_acc`, simplified
  from the old 4-parameter `a/b/c/f` form) and the optional Nienhuis/O'Brien
  estuarine blend, both now nested under `river_processing.
  empirical_estimation` and only relevant when `depth_method == "empirical"`.
- `testing` (rule "12") is now rule 11 (`11_testing.smk`, containing both
  `test_upstream_boundary` and `test_bifurcation_calibration_options`) —
  closing the numbering gap left by the consolidation above.

## New rules `08b`/`08c`: per-basin grid resolution + one shared SFINCS grid

Previously every basin used one fixed `sfincs.grid.resolution`, and rule 09
(conditioning), rule 10-modelled (calibration), and rule 13 (production
build) each independently let HydroMT resample onto their own grid.

- `08b_optimize_grid_resolution` (`src/grid_resolution.py`,
  `compute_optimal_resolution()`): resolution = `max(preferred_target,
  sqrt(domain_area / max_active_cells))` — preferred_target is either a
  flat default or, when `sfincs.grid.optimize_resolution.enabled` (default
  on), derived from the 20th-percentile channel width among
  discharge-thresholded reaches (`target_cells_per_width`). The
  `max_active_cells` cell-budget cap always applies as a safety net.
- `08c_build_sfincs_grid` builds the SFINCS regular grid once from that
  resolution and persists its transform/shape/CRS
  (`{basin_id}_sfincs_grid.json`) so rules 09/10-modelled/13 all target the
  identical grid instead of three independently-resampled, potentially
  pixel-misaligned copies.
- New `sfincs.grid.active_mask` (`enabled`, `elevation_buffer_m`): excludes
  cells far above anything the river ever reaches — ceiling = rule 09's own
  `river_elevation_max.json` + the buffer. Used by both rule 10-modelled's
  calibration and rule 13's production build.

## Modelled river-depth calibration: now confirmed working end-to-end

`river_processing.depth_method: "modelled"` (`src/river_depth_calibration.py`,
`10_depth_estimation_modelled.py`) runs a disposable, confined SFINCS
calibration model — round 0 isolates every reach behind an artificially
high (1000 m) wall to split the simulated water rise into channel
excavation + weir crest, then `n_correction_iterations` further rounds
re-run the model with the CURRENT excavation + real per-reach crest,
raising the crest wherever the model's own simulated water level still
overtops it, until convergence. Earlier this session this had several real
gaps; all are now fixed and verified (basin 2433835):

- Weir domain-edge/inactive-cell gaps, a native-burn-vs-`channel_mask`
  consistency constraint, and a subgrid phase-lock issue (the burn was
  landing on the wrong fine-pixel phase relative to the subgrid's own
  reference grid) — all three confirmed fixed via direct inspection of
  `dep_subgrid.tif` (leak reduced to a negligible 11/7671 residual pixels)
  and the network endpoint-degree graph (only 2 legitimate open-ocean
  boundary ends remain).
- **Convergence was only ever checked at centerline cells** — the seed
  reach's own array-edge/domain-boundary cells (and, structurally, the
  exact same blind spot already existed for the coastal probe mechanism,
  which updated its own crest every round but was never actually checked
  for convergence) could flood well beyond their nominal crest without the
  loop ever noticing, since the crest there comes only from dilation, never
  independently verified. Fixed by adding a new **river-boundary land probe
  set** (land cells within the dilation radius of the channel, mirroring
  the existing coastal-probe pattern exactly: same
  `_rasterize_nearest`/`_resolve_map_cell_idx`/`_read_map_zs_at_cells`
  machinery, same two-case crest-update formula) and requiring **all three**
  probe sets (centerline, coastal, river-boundary) to be genuinely
  contained before declaring convergence or accepting the final
  crest-tightening snap — not just the centerline gap as before. Any
  basin/reach whose buffered corridor reaches land beyond where its
  centerline was actually sampled (a clipped reach start, a sharp bend, a
  narrow headwater) is now protected the same way the coastal boundary
  already was.
- `check_convergence()` (the same windowed-flatness check) is now also
  reused, unmodified, by new rule 13b (below) — one shared convergence
  definition for both subsystems.

## New rule `13b_validate_protection_level`

Independent side check, not gating rules 14/16 and not gated by them:
re-simulates two short, steady discharge scenarios (protection-level design
discharge, and that discharge × `higher_rp_factor`) against the
ALREADY-BUILT scenario `default` production model — grid/elevation/
roughness/subgrid/weir referenced via relative paths, not rebuilt (same
technique `run_spinup` already uses) — to confirm the built model actually
holds at its own design protection standard rather than merely assuming
the calibration/weir stack produces one. New config section
`sfincs.protection_validation`.

## Performance fix: `05a_get_elevation`'s hard-merge fillnodata

Found while investigating why a basin 4267691 preprocessing run appeared
stalled for ~3 hours: the step-4 hard-merge's "defensive" `fillnodata()`
call (meant only for the rare case where neither FathomDEM nor GEBCO has
data for a pixel) was searching with `max_search_distance =
max(merged.shape)` — effectively the whole raster — because its target
mask (`np.isnan(merged)`) is computed *before* the `outside_domain` mask is
applied, and both sources are already NaN everywhere outside the delta
polygon. For a polygon that doesn't fill its own bounding box (the normal
case), that made the "rare" fallback run across most of the raster with an
unbounded search radius, only to have every one of those pixels discarded
one line later by `merged[outside_domain] = np.nan`. Fixed by capping
`max_search_distance` to a small constant (100 px) instead of the full
raster dimension — genuine small interior gaps still fill correctly from
nearby real data; the outside-domain area now costs a cheap bounded search
instead of an unbounded one. (Rejected alternative: assigning outside-
domain pixels a numeric sentinel instead of NaN before the fill — this
would make `fillnodata` treat them as valid *source* data and risks
blending a nonsense value into a genuine nearby gap's fill; NaN reliably
propagates instead.)

## Config renames / relocations

- `protection_levels` (top-level config section) renamed to
  `flopros_range` — the old name collided in spirit with the unrelated
  `protection_levels.json` rule-04 output and the `boundary_setup`
  section's own protection-adjacent keys. There is no `enabled` key on
  this section (never was one that did anything independent of the
  riverine/coastal netting logic already in rule 07).
- `boundary_setup` moved from a config.yml top-level section to
  `sfincs.boundary_setup` — it configures how rule 13 (a `sfincs`-family
  rule) consumes forcing, so it belongs alongside the rest of `sfincs.*`.

---

# 2026-07-20: scenario assessment + flood metrics table (- KL)

## Scenario axis ({scenario} wildcard, rules 13–17)

The pipeline can now run the same basin under multiple named flood-event
scenarios without re-running preprocessing (rules 01–12 stay scenario-free).

- New `config/scenarios.yml` (path set by new config key `scenarios_file`)
  defines named scenarios as (surge_rp, river_rp) pairs; a null RP uses the mean
  conditions for that driver. Example scenarios: baseline (no design event, mean conditions),
  coast_100 (100-yr RP coast, mean river discharge), river_100 (100-yr RP river discharge, mean coast),
  compound_100 (100-yr river discharge and coast).
- 00_common.smk loads and validates the scenario definitions (surge RPs must
  be COAST-RP tabulated values; river RPs in [2, 1000] yr), exposes
  `scenario_params(name)` and a `{scenario}` wildcard constraint. The
  reserved name `default` replays config.yml's own boundary_setup settings
  and is what plain `snakemake build` runs; other scenarios are selected via
  `--config target_scenarios="['baseline','coast_100']"`.
- All build-and-run outputs (rules 13–16) moved from
  `{basin_id}/sfincs|visuals/...` to
  `{basin_id}/scenarios/{scenario}/sfincs|visuals/...`.
  so that each scenario has its respective output visuals
- Rule 13 now rebuilds BOTH forcings per scenario without re-running rule 07:
  discharge via the existing `build_design_discharge_matrix()` (extended:
  `design_rp_yr=None` → constant bankfull hydrograph), and surge via new
  `surge.build_design_surge_matrix()` (`None` → flat baseline).
  To support this, rule 07's surge_forcing.nc now also stores the full
  COAST-RP table (`storm_tide_rp_table`), mirroring the `discharge_rp_table`.
  This way, RP's can simply be extracted from the corresponding tables before runs.
- The `boundary_setup.mode` enum validation in 00_common.smk was removed
  (commented out) — mode is now effectively always "compound" for named
  scenarios (i.e. always including both mean river discharge and mean coastal conditions),
  with per-driver nulls replacing coastal_only/river_only.

## New rule 17 (flood metrics calculations and tables)

Flood metrics are generated per scenario from rule 16's finished event run;
cheap postprocessing only, never re-runs SFINCS. Outputs under
`{basin_id}/scenarios/{scenario}/metrics/`: `max_flood_depth.tif` (downscaled
max-depth GeoTIFF, the input for later flood-source attribution and
adaptation measures) and `flood_metrics.csv` (one row of scalar metrics:
flooded/urban-exposed area, extent %, mean/max depth, volume — column names
match the legacy analyse.py risk_metrics.csv). New config keys:
`metrics.hmin` (0.05 m flood threshold) and `metrics.urban_landuse_code`
(50 based on landuse cover codes). Backed by new `postprocessing.compute_risk_metrics()`.

## Fixes

- `postprocessing.WATER_LANDUSE_CODES` corrected from (0, 200) to (80, 200) —
  code 80 ("Inland water") was intended all along per the adjacent comment;
  0 is the raster nodata value, so land-masking previously dropped nodata
  cells instead of permanent inland water.

---

# improve_base_model (merged as PR #1) (- JS)

Summary of functional changes on `improve_base_model` relative to the last
committed state (`5efcf4e`, "major updates"). This covers a large amount of
uncommitted work accumulated across many development sessions. Organized by
theme, not chronologically.

## River discharge design: return-period table instead of a single fixed value

Previously, EVA (rule 07) fit a POT/GPD curve and immediately extracted one
scalar flood discharge (`eva.rp_fl`) that got baked into the SFINCS discharge
timeseries at forcing-build time. Changing which return period to design
against meant re-running the whole EVA fit.

- Rule 07 now writes a full discharge table (`discharge_rp_table`, dims
  `crossing × return_period`) to `river_forcing.nc`, evaluated at a standard
  set of return periods (`STANDARD_RETURN_PERIODS_YR`: 1, 1.5, 2, then 5-yr
  steps to 1000, then 1000-yr steps to 10000) via new
  `gpd_return_value_table()` (`src/extreme_values.py`), a vectorized sibling
  of the existing `gpd_return_value()` — same fit, no re-fitting.
- The actual design discharge is now looked up **at SFINCS build time**
  (rule 13) via new `build_design_discharge_matrix()`
  (`src/river_forcing.py`): log-RP-interpolates from the table, applies the
  protection-discharge floor, and reconstructs the sinusoidal hydrograph.
  **Changing `boundary_setup.design_rp_river_yr` no longer requires
  re-running EVA — only rebuilding.**
- New config key `boundary_setup.design_rp_river_yr: 150` replaces
  `boundary_forcings.river.eva.rp_fl` as the production driver; `rp_fl` is
  still read internally but now only as a diagnostic RP for the EVA plot/CI.
- Three test scripts (`test_discharge_sensitivity.py`,
  `test_discharge_return_period_response.py`,
  `test_river_burning_methods.py`) now share `build_design_discharge_matrix`
  instead of each duplicating GPD-reconstruction logic.
- Step 4 (`visible_on_grid`, width-vs-grid-resolution check) in rule 07 is
  now informational only — it no longer gates GloFAS matching/EVA, so a
  narrow crossing at fine resolution can no longer strand and silently drop
  an entire downstream branch.
- The EVA diagnostic plot now picks the crossing with the highest `q_rp100`
  (previously: widest crossing by raw SWORD width, which didn't reliably
  track actual discharge magnitude) and reuses the actual (possibly
  bias-corrected) series instead of re-fetching raw GloFAS.
- Surge MDT vertical correction is now mandatory (the
  `vertical_correction.enabled` toggle was removed).
- `glofas_variable` config key removed — hardcoded to `"dis24"` (the sole
  variable in the catalogue's GloFAS v4 source).

## River DEM burning (new feature)

New optional step (`river_processing.burn_rivers.enabled`, requires
`conditioning.enabled`) that burns the river-bed profile directly into a
channel-only, native-resolution DEM, upstream of the SFINCS build:

- New `src/river_burn.py` (`burn_river_channel()`): processes each reach
  independently using only that reach's own `zbed_anchors` points and own
  full centerline. This works around a real hydromt_sfincs bug: its own
  `burn_river_rect` (subgrid `river_list=`) matches each subgrid tile
  against the *global, unclipped* `zbed_anchors` with no distance cutoff,
  causing cross-tile contamination (confirmed on basin 4267691's Mississippi
  headwater: a flat ~11.68 m profile became a wavy 12.6–14.8 m burned
  level).
- Junction-value blending (`_junction_value`): a synthetic boundary anchor
  at each reach's start/end, valued from the immediate up/downstream
  neighbour(s)' own nearest anchor (averaged at confluences/bifurcations),
  removing the old flat-clamp discontinuity where two reaches met.
  Extrapolation beyond a reach's own anchor range is clamped, not linear, to
  avoid overshoot.
- New rule `burn_river_dem` (`11b_burn_river_dem.smk` /
  `11b_burn_river_dem.py`), scheduled between `river_preburn` (11) and
  `testing` (12). Output `{basin_id}_river_burned_dem.tif` is registered in
  rule 13 as a higher-priority elevation source ahead of
  `elevation_merged`/`elevation_conditioned` (gaps fall back as before), and
  `burn_river_rect` is skipped entirely when this is enabled (no
  double-burning).

## River network processing overhaul

- **Bifurcation discharge splitting is now pure width-proportional.** The
  previous angle-weighted splitting (`_angle_factor`, `_end_direction_vec`)
  was removed entirely — it was never sensitivity-tested and added
  complexity without validated benefit.
- New `identify_delta_outflow_points()`: flags reaches that cross the delta
  polygon's *outline* (not filled interior) and aren't already a seed/mouth/
  bifurcation, returning their outline-crossing points. These become genuine
  SFINCS outflow boundary points (mask=3) via
  `mask_component.create_boundary(btype="outflow", ...)` in rule 13, instead
  of being silently discarded.
- New `remove_reaches_with_missing_width()`: must run *before* the
  width-order fix (else nodata sentinels get swapped into a real column);
  fills whichever of width/max_width is present from the other, and removes
  reaches where both are missing (plus their unbranched neighbour chain).
- New `enforce_mouth_width_monotonic()`: raises a river mouth's width to
  match its widest upstream neighbour if SWORD reports it narrower
  (treated as an imagery artifact, not real channel narrowing).
- `clip_anomalous_max_width()` (the `max_width_to_width_ratio` clip) was
  removed entirely from `normalize_channel_widths()`.
- New `enforce_mouth_depth_monotonic()` (`src/estuarine_depth.py`), applied
  unconditionally in rule 09b (even with the estuarine model disabled):
  floors a mouth's final depth at `max(own power-law estimate, upstream
  neighbour's depth)`. Motivated by basin 2433835 (Ebro), where the O'Brien
  mouth-depth estimate came out ~0.24 m vs ~2.3 m immediately upstream — a
  1.6 m sill sitting right at the model boundary.
- Rule 06 (`get_river_network`) no longer samples DEM elevation or enforces
  monotone downstream elevation on the raw network attribute — it is now
  just clip-to-domain + write. (Downstream monotonicity is still enforced,
  just later and more correctly — see rule 10's DEM conditioning, unchanged.)
- **Domain mode simplification**: the `"basins"` domain mode (bbox around
  intersecting HydroBASINS) was removed entirely. Domain is now always the
  delta polygon itself, reprojected to auto-detected UTM. Config keys
  `domain.mode`/`domain.buffer_m`/`domain.target_crs` are gone; rule 02
  still writes an `{basin_id}_intersecting_basins.gpkg` output (kept for
  schema compatibility) but it's now just a copy of the domain polygon with
  `n_intersecting_basins` hardcoded to 0.
- `terrain.work_res_m` config key removed — DEM/GEBCO merge working
  resolution is now auto-derived from FathomDEM's own native pixel size.
- New `river_network` data catalogue entry now points at the
  **manually-corrected** SWORD v17c export
  (`SWORD_global_v17c_unpublished_modified.gpkg`, layer `global_edges`) —
  width/main_side values hand-adjusted per-reach for bifurcation behavior.
  New sibling entry `river_network_original` (unmodified v17c) is used only
  by the new bifurcation-calibration test/rule (below), not by production.

## Upstream boundary check redesign (rule 12)

- Distance metric changed from a single kinematic formula to
  `min(kinematic_distance, attenuation_distance)`. New attenuation distance:
  friction-damped amplitude decay marching upstream reach-by-reach along the
  mainstem, `A(x) = A0·exp(-μx)` with reference velocity fixed at the mouth
  value (re-deriving it from locally-decaying amplitude would make the decay
  non-convergent). New config keys `channel_manning_n`,
  `amplitude_threshold_fraction` replace the removed
  `min_depth_at_mouth_m`/`min_river_velocity_ms` floors — depth/velocity are
  now used exactly as calculated; a mouth with no usable value is skipped
  and logged, not clamped.
- Each mouth now uses its own nearest CoastRP station's `rp_level` instead
  of the domain-wide max surge amplitude (stations can vary 5–10x across
  one basin).
- Now reads `river_network_estuarine.gpkg` (final hybrid depth) instead of
  `river_network_processed.gpkg` (power-law only) — fixes a bug where mouth
  depths varied wildly (1.89–17.80 m across 5 mouths of the same delta,
  should be ~5.04 m uniformly).
- New rule `test_bifurcation_calibration_options`
  (`12_testing.smk` / `12b_bifurcation_calibration_options.py`): rebuilds
  the cleaned network from raw SWORD for both `river_network_original` and
  `river_network` (the corrected version) and compares discharge
  partitioning at every bifurcation, one figure per bifurcation, in
  `visuals/bifurcation_calibration_options/`.

## Config/env-var portability for collaborators

- New environment-variable overrides in `00_common.smk`:
  `GCFM_RESULTS_DIR`, `GCFM_RAW_DATA_ROOT`, `GCFM_SFINCS_EXE` — override
  `results_dir`, the data-catalogue root, and the SFINCS executable path
  respectively, so a second machine/collaborator's local paths don't need
  hand-edits to tracked YAML after every `git pull`. Documented in a new
  "Local machine paths" section in `CONTRIBUTING.md`.
- New validation in `00_common.smk`: `burn_rivers.enabled` requires
  `conditioning.enabled` (raises `ValueError` otherwise).

## Config cleanup (`config/config.yml`)

- Removed dead/superseded keys: `river_processing.width_column` (canonical
  width is now always `"width"`), `river_processing.cleaning` (`snapping_
  tolerance_m` — confirmed unused; `max_width_to_width_ratio` — function
  removed), `domain.mode`/`buffer_m`/`target_crs`, `terrain.work_res_m`,
  `datum_correction.mdt_variable`, `boundary_forcings.surge.
  vertical_correction` (now mandatory), `boundary_forcings.river.
  glofas_variable`, `boundary_forcings.river.eva.rp_fl` (replaced by
  `design_rp_river_yr`), `sfincs.sanity_checks.velocity_animation` (feature
  removed, see below).
- Unified `boundary_forcings.surge.lead_days`/`dt_hr` and
  `boundary_forcings.river.lead_days`/`dt_hr` into a single top-level
  `boundary_forcings.lead_days`/`dt_hr`, shared by both.
- Unified `sfincs.spinup.timeout_s` (68400s) and `sfincs.event.timeout_s`
  (43200s) into a single `sfincs.simulation.timeout_s: 68400` (19h, the
  larger of the two), shared by both the spin-up and main event subprocess
  calls.
- Relocated `grdc_search_radius_km` from `boundary_forcings.river` to
  `boundary_forcings.river.bias_correction.grdc_search_radius_km` (it drives
  GRDC station matching, which is used both by the always-on correlation
  diagnostic and by the opt-in bias correction).
- Added `boundary_setup.design_rp_river_yr: 150`,
  `river_processing.burn_rivers.enabled: true`,
  `testing.upstream_boundary_check.channel_manning_n`/
  `amplitude_threshold_fraction`.
- A number of working/experiment value changes worth double-checking before
  treating as final production defaults: `sfincs.grid.resolution` 500→120,
  `sfincs.grid.quadtree.enabled` true→false, `sfincs.subgrid.
  nr_subgrid_pixels` 20→4, `sfincs.subgrid.nr_levels` 30→20,
  `boundary_forcings.surge.return_period` 100→25, `eva.peaks_per_year_min`
  3→1.0, `bias_correction.enabled` true→false, `profiling.enabled` true→false.

## Velocity animation feature removed

Out of scope for this project — removed entirely: `plots.animate_velocity`,
`postprocessing.compute_velocity_timeseries`, the `_ANALYSES` dispatch dict
and `postprocess_sfincs_output()` wrapper, the `animation_velocity` rule
outputs, and `sfincs.sanity_checks.velocity_animation` config. `storevel` is
now hardcoded to 0 in both rule 13 and rule 14 — nothing needs instantaneous
u/v output.

## Quadtree / big-domain memory fixes (`postprocessing.py`)

Several rounds of fixes to make big-domain quadtree postprocessing memory-
safe (all confirmed against real `MemoryError`/`ArrayMemoryError` crashes on
basin 4267691):

- `_mosaic_quadtree_dep_levels` now reads each per-level tif via a rasterio
  **decimated read** (`out_shape` + `Resampling.average`) at a shared,
  memory-bounded resolution, instead of reading every level at native
  resolution first.
- `_coarsen_for_memory` (renamed from `_coarsen_for_timeseries`) now chunks
  the array before `.coarsen().mean()` — an unchunked coarsen needed a
  comparably-sized temporary bookkeeping array on top of the already-huge
  source array.
- `write_crs`/`write_transform` calls switched to `inplace=True` (they
  deep-copy the whole array by default even though they only touch
  metadata).
- Animation no longer rasterizes quadtree runs at all — mesh data is
  rendered directly via `xugrid`'s own `.ugrid.plot()`/`PolyCollection`
  (memory scales with cell count, not a dense raster). `_rasterize_like()`
  was removed as dead code.
- New `compute_flood_timeseries_stats()` processes one timestep at a time
  (via `hydromt_sfincs.utils.downscale_floodmap`, discarding each frame
  before the next), decoupling memory cost from timestep count and allowing
  a more generous per-frame memory budget than the old animation-oriented
  shared budget.
- Fixed a real subprocess-timeout bug in `14_run_spinup.py`/`16_run_event.py`:
  `t_out.join()`/`t_err.join()` were called *before* `proc.wait(timeout=...)`,
  which blocks until SFINCS's stdout/stderr pipes close (i.e. until it has
  already exited) — silently making the configured timeout unreachable.
  Reordered so `proc.wait(timeout=...)` runs first.
- **Known discrepancy, not yet resolved**: `WATER_LANDUSE_CODES` changed
  from `(80, 200)` to `(0, 200)` at some point in this branch's history, but
  the surrounding comment/docstring still say `80 = Inland water` — this
  looks like it may be an unintentional edit rather than a deliberate
  change, since code `0` is not the documented "inland water" class in the
  Copernicus LC100 scheme this pipeline otherwise uses. Flagged for
  developer follow-up; left as-is for now.

## `plots.py` additions

- New `plot_global_protection_map()`: world choropleths (riverine/coastal
  FLOPROS protection RP by country), used by new standalone
  `tests/plot_global_protection_levels.py`.
- `_downsample_wgs_dataarray` replaced by `reproject_max_for_plot()`:
  reprojects directly at a coarse target resolution (avoiding a huge
  intermediate allocation) using max-resampling (avoiding aliasing away
  isolated peak values — important for a max-inundation map).

## Other SFINCS build fixes (rule 13)

- Fixed a real hydromt_sfincs quadtree bug: the spatially-varying initial
  water level (zsini) was silently never written for a quadtree build
  (`create()` sets config key `"ncinifile"`, but `SfincsQuadtreeGrid.write()`
  actually checks `"inifile"`). Worked around by setting both keys around
  the write call.
- Fixed `discharge_points.create()`'s region-membership check: its
  `buffer=` parameter does not expand the acceptance region outward
  (confirmed by reading hydromt_sfincs source). Crossings that fall just
  outside the exact unbuffered region are now explicitly snapped onto the
  boundary.

## Testing infrastructure

New: `tests/_snakemake_script_runner.py` (executes one `workflow/scripts/*.py`
"script:"-style file standalone with a mocked `snakemake` object, for
benchmark/sweep tests that reuse production build/run scripts verbatim
across many parameter combinations), `tests/test_bank_elevation_check.py` /
`test_bank_elevation_check_sfincs.py`, `tests/test_bifurcation_calibration_
options.py`, `tests/test_discharge_return_period_response.py`,
`tests/test_grid_resolution_benchmark.py`, `tests/test_river_burning_
methods.py`, `tests/plot_global_protection_levels.py`.

Removed: `tests/test_river_depth_sensitivity.py`,
`tests/plot_width_max_width_heatmap.py` (tied to the now-removed
`clip_anomalous_max_width`) — superseded in spirit by the new
benchmark/burning-methods tests.

## Reorganization

- `tools/` is a new directory for one-off/manual scripts (as opposed to
  pipeline tests): `download_fathomdem.py`, `download_glofas.py` (moved
  unchanged from `tests/`), `prepare_modified_sword_dataset.py` (moved from
  repo root).
- `Leuven_et_al/` (new, untracked): vendored reference material backing
  `src/estuarine_depth.py`'s O'Brien-relation implementation — not pipeline
  code.
- `hydromt_sfincs_subgrid_dtype_bugfix.py` (new, repo root): write-up +
  minimal repro of a memory bug in hydromt_sfincs's own
  `SubgridTable.write_netcdf()` (three `np.zeros()` calls default to
  float64 instead of matching the float32 arrays they're derived from,
  doubling peak memory) — intended for reporting upstream, not wired into
  the pipeline.
- Removed stale generated artifacts: `figs/dem_coverage/*`,
  `environment_diff.yml`, `updated_environment.yml`.

## GitHub collaboration tooling

`CONTRIBUTING.md`, `.github/workflows/ci.yml`,
`.github/pull_request_template.md`, and `tests/check_code_health.py`
(syntax + src-import check, no raw data needed) added for two-person GitHub
collaboration. `tools/sync_environment.py` + a local pre-commit hook keep
`environment.yml` in sync with the live `hmt_sfincs_dev` conda environment
automatically.
