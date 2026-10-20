# Consistency sweep (temporary)

Read-only except dimensions 3 (docs) and 4 (string literals). Base: `933017b` plus the
uncommitted refactors (#7, #13, #8). Line numbers are current. Format:
`file:line | issue | evidence | proposed fix | impact | status`.

## Needs decision

Behaviour (silent wrong or missing results):
- [x] **D1** `workflow/scripts/07_get_boundary_forcings.py:528`: a crossing is activated on a finite `q_rp2` alone. When the POT/GPD fit fails, it stays active with an all-NaN `discharge_rp_table`, so design discharge becomes NaN. `eva.ok` (`extreme_values.py:669`) is computed but unused. Fail the rule, or gate on `eva.ok`?
- [x] **D2** `workflow/src/extreme_values.py:476-481,614`: an exhausted threshold search keeps the last threshold, flagged `reject`. The flag reaches only a plot title (:712), but that GPD fit still feeds `discharge_rp_table`. Fail, or store the status per crossing and refuse `reject`?
- [x] **D3** `workflow/scripts/10_depth_estimation_modelled.py:707-709`: with no restart file, the steady-state loop breaks and calibration continues from a non-steady state (warning only). Raise?
- [x] **D4** About 10 branches write empty sentinel outputs (PNG/CSV/MP4/GPKG) when SFINCS outputs are missing, so the rule exits 0: `16_run_event.py:111-117,136-141`, `14_run_spinup.py:228-230,244-246,269-271`, `15_sanity_checks.py:58-60,99-101`, `10_depth_estimation_modelled.py:901-924`, `plots.py:2301-2306`. Raise instead? (keep the documented 0-byte `sfincs_subgrid.nc`/`sfincs.weir` placeholders)
- [x] **D5** Code defaults that disagree with config, decisive only if a key is removed: `rp_bf` 2 vs 1.5 (`extreme_values.py:511,687`, `07:567`); `bias_correction.enabled` True vs false (`extreme_values.py:227`); `grid.rotated` False vs true (`00_common.smk:90,98`). Make the keys required?
- [x] **D6** `deseasonalize_for_decorr` (`extreme_values.py:516`) is not in config, so the hard-coded True decides. Add it to `config.yml`?
- [x] **D7** Reach-ID normalisation (4 copies, edge-case semantics differ): already in TODO.md.
- [x] **D8** Discharge-point placement (3 copies): 10 raises when no points remain; 13/14 only warn, and keep all `dis_df` columns after filtering locations. One helper: should an empty result raise or warn?
- [x] **D9** `18a_adapt_pre.py:185-196`: sfincs.inp writer with different rules (also typo `scalar_ines`). Already in TODO.md.
- [x] **D10** MDT loading in `05a_get_elevation.py:306-336` vs `surge.load_mdt:420`: 05a sorts only lon, so its slice is empty on a descending-lat file. Move to `load_mdt` with a bbox?
- [x] **D11** Two UTM pickers: `geometry.pick_utm_crs` (centroid zone) vs `estimate_utm_crs` (`landuse.py:180`, `plots.py:2518`). They can differ near zone edges. Use `pick_utm_crs` everywhere?
- [x] **D12** `adaptation_scaling._read_polyline:68` vs `adaptation_method_pre._read_levee_trace:198`: different validation (CRS check vs empty/explode/LineString). Merge into one reader?

Config and data names (protected config / written outputs):
- [x] **D13** Unread config: `river_depth_modelling.gvf.target_discharge` (`config.yml:85`); catalogue `river_basins` (:29) and `deltadtm_mask` (:118); `measures.yml` `type`/`description`; `retreat.params.dep_subgrid` (always injected, never set); per-family `scales` override (`00_common.smk:281`). Delete or document?
- [x] **D14** `merge: false` in `adaptation_strategies.yml:36,39,60` is not declared in `measures.yml`, so it is never validated. Add a spec?
- [x] **D15** Channel Manning n has two keys: `testing.upstream_boundary_check.channel_manning_n` (0.030) and `gvf.manning_channel` (0.03). Merge?
- [x] **D16** Return-period data names (canonical `rp_yr`): scenario keys `surge_rp`/`river_rp`; `rp_fl`/`rp_bf`; the RP axis in netCDF (`return_period`/`table_rp`/`hg_rp`); `rp_level(_raw)`; `protection_rp`. Renaming changes config and output schemas.
- [x] **D17** Other data names without units or with two spellings: `baseline_m` vs `mdt` (and "baseline" = reference run); `_hr` vs `_hours` config suffixes; `bankfull_q`/`bankfull_discharge`/`mean_q`; `weir_crest_calibrated`; `rivdph` in our own gpkg; `water_level` vs `waterlevel` (`waterlevel_buffer_m`).
- [x] **D18** Tools/analysis-only code in `src`: `adaptation_lines` (tools only), `quicklook` (tests only), `extreme_values.plot_grdc_overview`, `plots.plot_global_protection_map`. Keep as library, or move next to their single caller?
- [x] **D19** Docs: README is empty (phase 8). README, CONTRIBUTING and CLAUDE.md demand an editable hydromt_sfincs clone, while env and CI pin `hydromt-sfincs==2.0.0rc2` from pip. CONTRIBUTING asks for Google `Args:`/`Returns:` docstrings, which contradicts the documentation standard. CLAUDE.md has two `<unknown: ask>` placeholders. Is `advance_1` the smallest adaptation case?
- [x] **D20** `plots.py:2082` legend "water level (period max)" is wrong for round 0 when `round0_level: end_of_run`. Which wording?

## 1. Dead functions, constants, CLI flags

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| workflow/src/raster.py:324 | `clip_ocean_from_topo` dead | 0 code refs (AST + git grep .py/.smk/Snakefile); only `data_catalogue.yml:140,166` notes mention it | delete; fix the catalogue notes (protected) | text | fixed |
| workflow/src/raster.py:271 | `reproject_to_reference_grid` dead | only caller is raster.py:341 (dead) | delete with the above | none | fixed |
| workflow/src/profiling.py:57 | `ScriptProfiler.time()`, `_WallTimer` (:85) and the stop() summary branch (:69-80) dead | 0 calls of `.time(` on a profiler | delete | none | fixed |
| workflow/src/gvf_target.py:648 | `GvfTargets.to_frame` dead | 0 refs (git grep -w) | delete | none | fixed |
| tools/build_esa_worldcover_vrt.py:33 | constant `TILE_RE` unused | 0 refs (AST) | delete | none | fixed |
| tests/unit/test_gvf_smoke.py:19 | constant `H_MIN` unused | 0 loads (AST) | delete | none | fixed |
| workflow/src/adaptation_lines.py (module) | public API used only by tools/build_adaptation_lines.py | no import from scripts/rules/Snakefile | D18 | none | fixed |
| workflow/src/quicklook.py:159 | used only by tests/unit/test_quicklook.py | no pipeline import | D18 (intended diagnostic) | none | fixed |
| workflow/src/extreme_values.py:1003 | `plot_grdc_overview` used only by tests/inspect_grdc_correlation.py | 0 production refs | D18 | none | fixed |
| workflow/src/plots.py:603 | `plot_global_protection_map` used only by tests/plot_global_protection_levels.py | 0 production refs | D18 | none | fixed |

No dead CLI flags: all 18 argparse dests are read. No dead module constants in src/scripts.

## 2. Config keys

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| config/config.yml:85 | `gvf.target_discharge` unread | git grep -w: only config.yml | D13 | none | fixed |
| workflow/src/extreme_values.py:511,687; scripts/07:567 | `rp_bf` default 2 vs config 1.5 | three sources of truth | D5 | behaviour if key removed | fixed |
| workflow/src/extreme_values.py:227 | `bias_correction.enabled` default True vs config false | removing the key silently enables GRDC quantile mapping | D5 | behaviour if key removed | fixed |
| workflow/rules/00_common.smk:90,98 | `rotated` default False vs config true; 18a/18b/18c comments say "defaults to true" | code vs docs | D5 | behaviour if key removed | fixed |
| workflow/src/extreme_values.py:516 | `deseasonalize_for_decorr` absent from config | 0 hits in config/ | D6 | none | fixed |
| workflow/rules/00_common.smk:259 | `.get("scales", [0.4, 0.9, 1.0])` copies adaptation_strategies.yml:18 | same value twice | index directly | none | fixed |
| workflow/rules/00_common.smk:281 | per-family `scales` override undocumented, unused | no family sets it | D13 | none | fixed |
| config/data_catalogue.yml:29,118 | `river_basins`, `deltadtm_mask` unreferenced | 0 hits; `river_basins_path` param left at plots.py:242,261 | D13 | none | fixed |
| config/measures.yml:7-8 | `type`/`description` unread | readers use only params.{min,max,scale,required_for} | D13 | none | fixed |
| config/measures.yml:164 | `retreat.params.dep_subgrid` declared but always injected | 18a_adapt_pre.py:90-92; path resolution 00_common.smk:368 dead | D13 | text | fixed |
| config/adaptation_strategies.yml:36,39,60 | `merge` param not declared, never validated | validate_measure_params loops over spec keys only | D14 | none | fixed |
| workflow/src/river_forcing.py:787 | `event_hydrograph.source` typo silently gives the cosine wave | no check of allowed values | raise unless in {glofas_event, cosine} | behaviour (typo) | fixed |
| workflow/rules/00_common.smk:379; scripts/17_flood_metrics.py:34 | refer to measure `water_retention_greening`, which is not defined | commented out elsewhere | drop references | none | fixed |
| workflow/rules/00_common.smk:271 | excludes a `meta` key that does not exist | yml has only scales + families | drop | none | fixed |
| workflow/rules/00_common.smk:239 | `.get("adaptation_measures", {})` on a whole section | a misnamed key gives a misleading "measure not defined" later | index directly | none | fixed |
| workflow/src/gvf.py:111,134-148; gvf_target.py:710-766; extreme_values.py:230-231,433-439,507-515; scripts/08:151-153; profiling.py:24-27 | about 40 `.get` defaults copy config values | two sources of truth (equal today) | read keys directly, validate at boundary | none | fixed |
| workflow/scripts/10_depth_estimation_modelled.py:1135 | `steady_cfg.get("fail", False)` while siblings are indexed strictly | inconsistent | index | none | fixed |
| workflow/rules/00_common.smk:163,184-185 | scenario `discharge_multiplier`/`slr_m` defaults never used | all 9 scenarios set both | require, or document as optional | none | fixed |
| workflow/rules/00_common.smk:51,215,391,404 | CLI-only keys `target_*` absent from config.yml | defaults decide (all basins, ["default"], [], ["pre","post"]) | list commented in config.yml / README | none | fixed |
| config/data_catalogue.yml | per-dataset fields other than name/file_path/attributes never read | 0 reads of crs, provider, ... | say "documentation" in the header | none | fixed |
| config/config.yml:217 vs :86 | channel Manning n under two keys | 0.030 vs 0.03 | D15 | none | fixed |
| workflow/scripts/18a_adapt_pre.py:95 | `in ("water_retention")` is a substring test, not a tuple | missing comma | `in ("water_retention",)` | none (today) | fixed |

## 3. Docs vs code

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| CONTRIBUTING.md:11-12 | stages "00-16" | rules run to 19 | "through 19_*.smk" | text | fixed |
| CONTRIBUTING.md:19 | example "rule 17" for a new rule | 17 exists | "rule 20" | text | fixed |
| CONTRIBUTING.md:156-158 | PR checklist lacked pytest, check_code_health, pyright | ci.yml runs them | added | text | fixed |
| CONTRIBUTING.md:181-182 | says only pydocstyle skips workflow/scripts | global pre-commit exclude | every hook skips it | text | fixed |
| CLAUDE.md:14 | preprocess "rules 01-12" | no rule 12 | "rules 01-11" | text | fixed |
| README.md:1-13 | front door empty | setup stub only | D19 / phase 8 | text | fixed |
| README.md:9-13; CONTRIBUTING.md:202-213; CLAUDE.md:8 | editable hydromt_sfincs clone required | env and CI pin 2.0.0rc2 from pip | D19 | text | fixed |
| CONTRIBUTING.md:176-181 | asks for Google `Args:`/`Returns:` | documentation standard forbids repeating hints | D19 | text | fixed |
| CLAUDE.md:25,29 | `<unknown: ask>` placeholders | unresolved | D19 | text | fixed (adaptation case set; "Other protected paths" left for the owner) |
| workflow/Snakefile:172 | preprocess docstring "rules 01–12" | no rule 12 | "01–11" | text | fixed |
| .github/pull_request_template.md:4,17-21 | example "rule 09a"; no pytest item | no rule 09a; CI runs pytest | fix example, add item | text | fixed |
| CONTRIBUTING.md:35-49,62-63,324-326 | history and status wording, current branch names | documentation standard | trim (phase 5) | text | fixed |
| docs/gvf_discovery.md:416-424 | session record says there are no tests or CI tests; names deleted script | contradicted by tests/unit, ci.yml | gitignored record: leave or delete | text | open (now visible as untracked; not committed) |

## 4. String literals

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| workflow/scripts/09b_grid_align_landuse.py:80 | log says nearest-neighbour | :67 `Resampling.mode` | "(mode, single pass)" | text | fixed |
| workflow/src/plots.py:853,889,1043 | "GOCO6s" | GOCO06s everywhere else | "GOCO06s" (column width kept) | text | fixed |
| workflow/src/plots.py:2082 | legend "water level (period max)" wrong for round 0 | round 0 passes end-of-run zs by default | D20 | text | fixed |
| workflow/src/adaptation_scaling.py:1 | docstring names `strategy_def_for()` | 0 definitions | docstring fix | text | fixed (`strategy_def_for` now exists, 00_common.smk:346) |

Checked about 1,900 literals: only the rows above have stale names or values. Parity: changed files are AST-identical with string constants blanked; ruff unchanged; 219 passed.

## 5. Duplicated helpers

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| src/river_network.py:52; src/river_forcing.py:36; scripts/08:27; scripts/13_build_sfincs_skeleton.py:435 | 4 reach-id normalisers | differ on NaN, "nan", inf, "1e10" | D7 → `river_network.normalize_reach_id` | behaviour | fixed |
| scripts/10:566-588; scripts/13_build_sfincs.py:268-302; scripts/14:152-175 | discharge-point placement x3 | same 5 calls; 10 raises, 13/14 warn and keep all dis_df columns | D8 → `sfincs_run.place_discharge_points` | behaviour | fixed |
| scripts/18a_adapt_pre.py:185-196 | hand-written sfincs.inp | different exclusion rules | D9 → `sfincs_run.write_sfincs_inp` | behaviour | fixed |
| scripts/05a:306-336 vs src/surge.py:420 | MDT loading | lat sort differs | D10 → `surge.load_mdt(bbox=...)` | behaviour (descending lat) | fixed |
| src/geometry.py:11 vs landuse.py:180, plots.py:2518 | two UTM pickers | centroid vs area-of-use | D11 → `geometry.pick_utm_crs` | behaviour (zone edge) | fixed |
| scripts/13_build_sfincs.py:44-53; 13_build_sfincs_skeleton.py:42-52 vs src/log.py:8 | hand-rolled logging setup | no `force=True`; format differs | `setup_logging(verbose=True)` | text | fixed |
| scripts/05a:241-275 vs src/raster.py:100 | tile merge x2 | same merge; raster.py lacks try/finally | `raster.merge_tiles` | none if parity | fixed |
| src/gvf.py:26 vs src/river_depth_calibration.py:34 | seed-discharge table x2 | same core, different q and mask | `river_forcing.seed_discharge_by_reach` | none | fixed |
| src/adaptation_scaling.py:68 vs adaptation_method_pre.py:198 | polyline readers x2 | different validation | D12 | behaviour | fixed |
| src/adaptation_method_post.py:24,778 vs pre:158,1277 | measure framework copied | AST-identical | new `src.adaptation_measures` | none | fixed |
| scripts/10:356-386 vs 13_build_sfincs_skeleton.py:240-270 | sea + outflow boundary block | line-for-line | `sfincs_run.set_sea_and_outflow_boundaries` | none | fixed |
| scripts/10:339-346 vs 13_build_sfincs_skeleton.py:221-228 | active-mask elevation ceiling | same 6 statements | `sfincs_run.create_active_mask` | none | fixed |
| scripts/10:358,391,432; 13_build_sfincs_skeleton.py:241,287; tools/build_adaptation_lines.py:71 | landuse_on_grid read + shape check x6 | same ValueError text | `raster.read_on_grid` | none | fixed |
| tools/build_adaptation_lines.py:205 vs src/grid.py:144 | grid JSON loaded by hand | no Affine, no shape | `grid.load_grid_def` | none | fixed |
| scripts/15:64-81 vs scripts/16:76-93 | inundation-ratio block | block match incl. broad except | `postprocessing.inundation_stats` | none | fixed |
| src/river_burn.py:236-295,480-497 vs river_conditioning.py:164-204 | along-line profile and window-cell code | same interp/mask/xy steps | `river_network.clamped_profile`, `line_window_cells` | none | fixed |
| src/plots.py:116 vs src/grid.py:47 | 4-corner footprint | identical math | `grid.footprint(...).bounds` | none | fixed |
| 7 places (05a:312, raster.py:454, surge.py:429,455, river_forcing.py:184,350, 07:439) | lat/lon dim detection, 2 styles | | `raster.latlon_dims` | none | fixed |
| scripts/13_build_sfincs.py:385-434 | inline applied-forcing figure | src.plots has the source-forcing figure | `plots.plot_applied_forcing` | none | fixed |
| 10 direct `fig.savefig` calls (07, 07c, 11, 11b, 13, 13sk, tools, quicklook) | bypass `plots.save_figure` retry/mkdir/close | grep | `plots.save_figure` | none | fixed |
| src/adaptation_scaling.py:49 vs tests/check_crest_calibration.py:172-183 | sfincs.weir parser x2 | same loop | `sfincs_run.read_weir_file` | none | fixed |
| rules/18a_adapt_pre.smk:59-89,109-119 vs 13_build_sfincs.smk:19-53, 16_run_event.smk:23-32 | rule params copied | 0 differing keys | `00_common` param helpers | none | fixed |
| scripts/07:104-112 vs 07a:25-32 | surge station selection run twice | same call and params | 07 reads 07a's surge_stations.json | none | fixed |
| tests/ analysis scripts | copied helpers and preambles | AST-identical `_first_existing`, `label` | shared helper module once `analysis/` exists | none | fixed |
| tests/KL_flood_maps.py:73-79 vs plots.py:2183-2192 | depth panel re-implemented | block match | call `plot_max_inundation_map` | none | fixed |
| src/gvf.py:191 vs river_depth_calibration.py:171 | listed in AUDIT 12 as duplicate | different inputs, chosen per method | not a duplicate: close | none | resolved |

## 6. Error handling

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| scripts/07:528; src/extreme_values.py:663-669 | crossing active on finite q_rp2 alone | `eva.ok` unused; GPD fail → NaN table → NaN design_q | D1 | behaviour | fixed |
| src/extreme_values.py:476-481 | rejected threshold used silently | status only in plot title | D2 | behaviour | fixed |
| scripts/10:707-709 | non-steady state accepted | warning + break | D3 | behaviour | fixed |
| src/extreme_values.py:587 | broad except on GEV fit: missing lmoments drops every crossing, rule exits 0 | warning only | catch numeric errors; fail if 0 crossings fit | behaviour | fixed |
| src/extreme_values.py:374,412,661 | `except Exception: pass` (bootstrap, KS) | body `pass` | narrow; warn on NaN CI | behaviour (diagnostic) | fixed |
| src/postprocessing.py:273 | `except Exception` → pixel area NaN → area/volume columns NaN | fallback np.nan | remove try | behaviour | fixed |
| scripts/15:69-70; scripts/16:81-82 | same res → NaN pattern | log line only | remove try | text | fixed |
| scripts/14:228-271; 15:58-101; 16:111-141; 10:901-924; plots.py:2301-2306 | empty sentinels on missing SFINCS output, rule exits 0 | `.touch()` / empty DataFrame | D4 | behaviour | fixed |
| scripts/15:47-49,91-93 | 0-byte sfincs_map guard is dead | rule 14 never touches the map | delete | none | fixed |
| scripts/13_build_sfincs_skeleton.py:278,306-315 | dead `weir_grid = None` and unreachable else | weir_grid set at :280 | delete | none | fixed |
| scripts/13_build_sfincs_skeleton.py:302-303; scripts/07:358-360 | 0-byte .gpkg / .nc are invalid files | `.touch()` | write empty valid files | behaviour (latent) | fixed |
| scripts/14:239 | broad except around time decoding | re-opens with decode_times=False | catch ValueError | none | fixed |
| src/plots.py:300 | save_figure retries on every OSError | lock errnos are transient, ENOSPC is not | retry only EINVAL/EACCES | none | fixed |
| src/plots.py:2519; extreme_values.py:715,731 | broad except, harmless fallback / text in figure | | narrow | none / text | fixed |
| tools/download_glofas.py:50-53; download_cds_waterlevel.py:79-82; prepare_modified_sword_dataset.py:116-134; plot_dag.py:28-39 | tools exit 0 after failures | failures printed or skipped | exit non-zero | behaviour (tool) | fixed |

## 7. Naming (one concept, several names)

| file:line | issue | evidence | proposed fix | impact | status |
|---|---|---|---|---|---|
| config/scenarios.yml:37; config.yml:183; src/river_forcing.py:388; scripts/07:130 | return period (data names) | surge_rp 55, river_rp 43, rp_fl 14, rp_bf 11; RP axis return_period 41 / table_rp 13 / hg_rp 8; rp_level 31 | D16 → `*_rp_yr`, one axis `rp_yr` | behaviour (schema) | fixed |
| rules/13_build_sfincs.smk:35; src/surge.py:136 | return period (params, locals) | design_rp_surge_yr 17, design_rp_yr 20, rp 52, rps 19, table_rps 21 | `surge_rp_yr`/`river_rp_yr`; locals `rp_yr`/`rps_yr` | none | fixed |
| tests/unit/test_protection_levels_smoke.py:52 | protection RP | riverine_rp_yr 11 vs river_*; protection_rp 6 (no unit) | `river_rp_yr`, `protection_rp_yr` | behaviour if column renamed | fixed |
| src/surge.py:1 | calm-sea datum vs reference run | baseline_m 84, mdt 27+114, baseline* 346 | D17: datum `calm_sea_level_m`; keep "baseline" for reference run | behaviour (netCDF var) | fixed |
| config/config.yml:112 | time suffixes | _hr 269, _hours 77, trstout_sec | D17: `_hr`, `_s` | behaviour (config) | fixed |
| src/gvf.py:36 | bankfull / mean discharge | bankfull_q 22, bankfull_discharge 13, q_bf 7, mean_q 20, mean_discharge 17 | D17: `*_discharge_m3s` | behaviour (netCDF/config) | fixed |
| scripts/10:864 | crest | crest 152, crest_elevation_m 24, weir_crest_m 13, weir_crest_calibrated (column) | D17: `crest_elevation_m`, column `_m` | behaviour (gpkg column) | fixed |
| scripts/10:862 | river depth | rivdph 45 (hydromt name in our gpkg), river_depth_m 2 | D17: rivdph only at the hydromt call | behaviour if column renamed | fixed |
| config/config.yml:229 | water level spelling | water_level 41, waterlevel 37, wl 24 | D17: `water_level_m`; keep zs/zsmax/GTSM at I/O | behaviour (config key) | fixed |
| rules/08c_build_sfincs_grid.smk:16 | grid spacing | resolution 120, dx 58, cell_size_m 20, work_res_m 9 | `resolution_m` params, `dx_m` in grid math | none | fixed |
| src/adaptation_method_pre.py:31; src/gvf_solver.py:1 | bed / water level / depth | dep 202, zb 20, elevation 247, dem 48, z 94, h_max 21, eta | `bed_elevation_m`, `water_level_m`, `depth_m`; keep dep/zb at SFINCS I/O | none | fixed |
| src/river_network.py:63 | reach id | reach_id 196, rid 193, rids 26 | `reach_id(s)` | none | fixed |
| src/protection_weir.py:43 | sea land-use code | SEA_CODE 18, LANDUSE_SEA 17, literal 200 | one `LANDUSE_SEA_CODE` in landuse.py | none | fixed |
| scripts/05a:464 | nodata | NODATA=-9999 defined in 05a, 05b, 10; literal -9999 51x/24 files | one `FLOAT_NODATA` in src.raster | none | fixed |
| config/config.yml:19 | model area | basin_id 760, basin 173, domain 341, delta 102 | basin_id=id, domain=model polygon, delta=source polygon | none (folders/config: decide) | fixed |
| config/data_catalogue.yml:495 | roughness vs Manning | roughness 125, manning(_n) 45 | `manning_n` for the coefficient | none | fixed |
| src/river_network.py:1 | width without unit | width 325, width_m 7, rivwth 6 | `width_m`; keep rivwth at hydromt I/O | none | fixed |
| src/surge.py:500; scripts/07:266 | SLR and time locals without unit | local_slr, global_mean_slr, slr_offset; times_s/_hours mix | add `_m` / `_hr` | none | fixed |
