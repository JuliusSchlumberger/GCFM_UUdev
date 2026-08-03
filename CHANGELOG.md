# Changelog

Newest changes first. See `Reference_memory.txt` for the current, up-to-date
description of how the pipeline works; this file only describes *what changed
and why*.

#

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
