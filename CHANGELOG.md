# Changelog

Newest changes first. See `Reference_memory.txt` for the current, up-to-date
description of how the pipeline works; this file only describes *what changed
and why*.

#

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
