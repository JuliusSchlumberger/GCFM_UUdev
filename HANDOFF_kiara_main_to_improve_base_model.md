# Moving from `main` to `improve_base_model`

For Kiara. This compares `origin/main` (10a6d0f, PR #16 merged) with `improve_base_model` (HEAD b084547 plus uncommitted working-tree changes).

Your `adaptation_modelling` branch (fe7b593, PR #17) is merged in, but rewritten for the new layout. Changes that came from your branch are marked **[KL]**, so you can skip them.

What changed and why is in `CHANGELOG.md`. Entries that touch the adaptation module start with a **"KIARA, PLEASE REVIEW"** block; the top two need your review.

---

## 1. Checklist to get running

1. **Site settings replace environment variables.**
   - Copy `config/site.example.yml` to `config/site.yml`. It is gitignored.
   - Fill in `results_dir`, `raw_data_root`, `sfincs_exe` and `adaptation_data_root` (the last is only needed for `adapt`).
   - Your old `GCFM_RESULTS_DIR`, `GCFM_RAW_DATA_ROOT` and `GCFM_SFINCS_EXE` user variables still override `site.yml`, without any warning. Delete them or point them at the same places. `docs/install.md` "Site settings" shows how.
2. **Conda env.**
   - The env is now named `gcfm`. `environment.yml` lists only the direct dependencies, and its pins match your current `hmt_sfincs_dev`, so that env keeps working.
   - Do not use `environment.win-64.lock.yml` yet: it is committed empty (`dependencies: []`, TODO S5). The pre-commit hook `sync-environment-lock` fails until an env named `gcfm` exists.
3. **New and changed raw data**, under `raw_data_root`:
   - **SWORD v17c, published release:** one GeoPackage per region, `SWORD/v17c/sword_{AF,AS,EU,NA,OC,SA}_v17c.gpkg`, layer `reaches`. It replaces `SWORD/SWORD_global_v17c_unpublished_modified.gpkg`. Download it by hand. The 9 manual width fixes now live in the repo, in `config/sword_width_corrections.csv`.
   - **GTSM station subset:** `GTSM_storm_tide_hourly/station_subset.nc`, required by rule 07b. Build it once, and again whenever a basin or domain changes:
     ```
     snakemake --until select_surge_stations --cores 4
     python tools/subset_gtsm_stations.py
     ```
   - **No longer needed:** `DeltaDTM_masks/deltadtm_mask.vrt`, `HydroBASINS/hybas_global_merged_lev06.shp`, the unpublished SWORD files.
   - **On Linux only:** the `AVISO/` folder is now spelled `Aviso/`.
4. **Adaptation geometries move to one folder per basin.** For the Ebro, `<raw_data_root>/adaptation_ebro/*.geojson` becomes `<adaptation_data_root>/2433835/`:

   | file | strategy / measure |
   |---|---|
   | `offshore_island.geojson` | `advance` / offshore_barrier |
   | `advance_pumps.geojson` | `advance` / pumps |
   | `land_reclamation.geojson` | `NbS_protect_open` / nbs_land_reclamation |
   | `water_retention_area_river.geojson` | `NbS_protect_open` / water_retention |
   | `water_retention_inlet.geojson` **[KL]** | `NbS_protect_open` / water_retention `inlet`, resolved like `locations` |

   - The `auto/...` lines are no longer input files:
     - `grey_protect_open` uses `coastal_protection_weir_PO` and `river_protection_weir_PO`;
     - `protect_closed` uses `coastal_protection_weir_PC` and `pump_river_mouth`.
   - Rule 12c `build_adaptation_lines` now builds them per basin, into `<results_dir>/{basin_id}/adaptation_lines/`. It runs automatically for any strategy that references `auto/<name>`. `tools/build_adaptation_lines.py` is gone.
5. **Rerun everything, from rule 02 on, for every basin.**
   - The SWORD lines are now oriented downstream.
   - The rule 11 calibration now holds the sea at the storm tide.
   - The attribution mask's class 4 changed (section 5).

   `python tools/list_stale_figures.py` lists figures left in the old folders; add `--delete` to remove them. It never touches data files.

---

## 2. Config changes

### `config/scenarios.yml`

- **Renamed keys in every scenario:** `surge_rp` -> `surge_rp_yr` and `river_rp` -> `river_rp_yr`. The old names read as null, and the scenario then fails as "both drivers null".
- **Removed:** `discharge_multiplier`.
- **Now required:** `slr_m`.
- **[KL]** Scenarios:
  - `mean_03SLR` and `storm_gloria` added;
  - `compound_100` renamed `compound_100_03SLR`;
  - `coast_only_100`, `river_only_100` and `compound_500` removed.
- **Attribution (rule 17a)** needs two counterparts for each scenario, both with the same `slr_m`:
  - the same `river_rp_yr` with `surge_rp_yr: Tide`;
  - the same `surge_rp_yr` with `river_rp_yr: Mean`.

  A missing counterpart is now an error.

### `config/adaptation_strategies.yml` and `config/measures.yml`

- **Value tokens renamed:** `max_coastal_waterlevel` -> `max_coastal_water_level` and `max_river_waterlevel` -> `max_river_water_level`. Only the new names are accepted (`workflow/src/adaptation_strategies.py:30`). Update any strategy file of your own.
- **Paths:**
  - `locations: adaptation_ebro/<f>.geojson` -> `"{basin_id}/<f>.geojson"`;
  - `adaptation_ebro/auto/<x>.geojson` -> `auto/<x>`.
- `config/adaptation_strategies_manual.yml` is deleted.
- `measures.yml`:
  - `coastal_levee` and `river_levee` gain a `merge` bool. `false` replaces the matched baseline weir segments; `true` appends to them.
  - `retreat.dep_subgrid` is removed; it now comes from the skeleton.
  - The per-measure `type` and `description` fields are removed.

### `config/config.yml`, the keys you are most likely to touch

| old | new | note |
|---|---|---|
| `animation_scenarios`, `timeseries_scenarios` **[KL]** | `analysis: true/false` (required bool) | `analysis` gates the animations, `flood_timeseries.csv` and the diagnostic figures; also `--config analysis=True` |
| (none) | `resources.threads.<rule>` | threads per rule, required for every rule |
| (none) | `adaptation.lines.*` | settings for rule 12c (`direction_length_m`, `max_tilt_deg`, `max_half_length_m`, `ext_cells`, `pump_upstream_m`) |
| `sfincs.slr_ramp_hours` | `sfincs.slr_ramp_hr` | all `*_hours` ramps are now `*_hr` |
| `sfincs.spinup.boundary_ramp_hours` | `sfincs.spinup.boundary_ramp_hr` | |
| `sfincs.spinup.spinup_days` 1 | 2 | must equal `boundary_forcings.lead_days` (also 2); the workflow refuses to start otherwise |
| `boundary_setup.waterlevel_buffer_m` | `water_level_buffer_m` | |
| `surge.storm_tide.ramp_hours`, `river.min_ramp_hours` | `ramp_hr`, `min_ramp_hr` | |
| `river.glofas_min_mean_discharge` | `river.glofas_min_mean_discharge_m3s` | |
| `river.eva.rp_fl`, `river.eva.rp_bf` | `river.eva.preview_rp_yr`, `river.eva.bankfull_rp_yr` | |
| `river.max_rise_per_hr` 0.05 | 0.02, plus new `river.max_rising_days: 7` | the river event starts from its real rising limb; event runs go from 120 to 228 h |
| `sfincs.grid.rotated`, `rotated_overrides` | removed | the grid is always axis-aligned |
| `river_depth_modelling.discharge_ramp_hours` | `river_depth_modelling.ramp_hr` | calibration; plus new `bed.*`, `verify_crests`, `depth_scale` and other keys |

The other new blocks are on the river side: `landuse.water_code`, `landuse.sea_max_elevation_m`, `river_processing.water_surface.*`, `width_qa`, `canal_routing` and `channel_manning_n`. The defaults are fine for adaptation work.

### `--config` flags

- `target_basins`, `target_scenarios`, `target_strategies` and `target_method` are unchanged.
- `analysis=True` is new.
- `animation_scenarios` and `timeseries_scenarios` **[KL]** are gone.
- A bare `snakemake` now runs `all` (it used to run `adapt`).
- The targets `preprocess`, `build`, `all` and `adapt` are unchanged.
- Jobs that read large raw files share the resource `raw_io`, which the profile `workflow/profiles/default/config.yaml` limits to 1. Override it with `--resources raw_io=N`.

---

## 3. Rules and scripts, renumbered in run order

Rule names are mostly unchanged; the file numbers changed. In particular, the adaptation rules keep their names.

| old file | new file | note |
|---|---|---|
| 01_separate_deltas + 02_modeldomain | 01_domain (`determine_model_domain`) | merged |
| 06_river_network | 02_river_network | `plot_river_network` moved to `analysis/plot_river_network.py` |
| 04_protection_levels | 03_protection_levels | |
| 08c_build_sfincs_grid | 04_sfincs_grid | |
| 02b_prepare_landuse + 03_land_polygons | 05_landuse_source (`prepare_landuse`) | merged |
| 05_static_terrain_data (elevation, land use) | 06_elevation (`get_elevation`) | merged; `sea_mask.tif` dropped |
| 07_boundary_forcings | 07d_boundary_forcings | 07a/b/c unchanged |
| 05c roughness + 09b_grid_align_landuse | 09_land_surface (`grid_align_landuse`) | merged |
| 09_condition_elevation | 10_river_surface (`river_surface`) | replaced |
| 10_depth_estimation_modelled | 11_depth_estimation_modelled | |
| 11_testing | `analysis/check_upstream_boundary.py`, `analysis/check_bifurcation_options.py` | no longer rules |
| 13_build_sfincs_skeleton | 12a_build_sfincs_skeleton | |
| (none) | 12b_land_bed_level | new; writes `sfincs_skeleton/land_bed_level.nc`, also inside the adapted skeleton |
| `tools/build_adaptation_lines.py` | 12c_build_adaptation_lines | **[KL]** tail cut and fallback are in `workflow/src/adaptation_lines.py` |
| 14_run_spinup, 15_sanity_checks | 13a_run_spinup, 13b_sanity_checks | |
| 13_build_sfincs | 14_build_sfincs | |
| 16_run_event | 15_run_event | **[KL]** edits ported |
| 17_flood_metrics | 16_flood_metrics | |
| 18c_attribution_mask | 17a_attribution_mask | |
| 18a_adapt_pre | 17b_adapt_pre | **[KL]** chained subgrid measures, inlet |
| 18b_adapt_post | 17c_adapt_post | |
| 19_adapt_compare | 18_adapt_compare | |
| (none) | 19_analysis_figures | plot-only rules that redraw figures from saved outputs; run with `analysis=True` |

Log files follow the new numbers (e.g. `logs/{basin}/runs/{sc}/15_run_event.log`).

---

## 4. Stand-alone scripts: `tests/` becomes `analysis/`

- All figure and check scripts now live in `analysis/`, without the `KL_` prefix. Shared helpers are in `analysis/_common.py`.
- They write to `<results_dir>/<basin>/figures/analysis/90_<script>_<what>.*`. Cross-basin figures go to `<results_dir>/figures/analysis/`.
- Your scripts still take `BASIN_ID` / `SCENARIO(S)` / `STRATEGY` from constants at the top of the file. The others take an optional positional `[basin_id]` (default 2433835).

| old (main) | your branch name | now |
|---|---|---|
| `tests/KL_flood_maps.py` | `flood_maps_attribution.py` | `analysis/flood_maps.py` (scenarios now `*_100_03SLR`) |
| `tests/KL_domain_figures.py` | `study_area_figures.py` | `analysis/domain_figures.py` (default basin now 2433835) |
| `tests/KL_results_figures.py` | `pre_post_scale_response.py` | `analysis/results_figures.py` |
| `tests/KL_csi.py` | `pre_post_heatmap.py` | `analysis/pre_post_heatmap.py`; `csi_pre_vs_post.py` removed |
| `tests/KL_flood_map_pre_post.py` | (deleted) | `analysis/pre_post_floodmap.py` |
| `tests/KL_analysis_waterlevel_animation.py` | `waterlevel_animation.py` | `analysis/water_level_animation.py` |
| `tests/KL_check_adaptation_scaling.py` | same | `analysis/check_adaptation_scaling.py` (argparse instead of `sys.argv`) |
| (none) | `hmin_sensitivity*.py` | `analysis/hmin_sensitivity.py`, `analysis/hmin_sensitivity_fig.py` |
| `tests/check_crest_calibration.py`, `inspect_grdc_correlation.py`, `plot_global_protection_levels.py`, `plot_overtopping_volume.py`, `plot_surge_vs_dike_crest.py` | | `analysis/`, same names |
| `tests/KL_gtsm_storm_tide.py`, `check_surge_shape_duration.py`, `plot_calibration_round_profiles.py`, `plot_coastal_protection_weir.py`, `tests/archive/` | | deleted; the rules draw these figures now |
| `tools/replot_event_figures.py` | | deleted; replaced by the rule `analysis_run_event` |
| `tools/prepare_modified_sword_dataset.py`, `tools/build_deltadtm_mask_vrt.py` | | deleted |
| repo root: `Reference_memory.txt`, `hydromt_sfincs_subgrid_dtype_bugfix.py`, `Leuven_et_al/`, `qgis/`, `figs/dag/*.svg` | | deleted; methods are in `docs/methods.md`, the hydromt fix is in `workflow/src/hydromt_sfincs_patch.py` |

**New tools:**
- `tools/subset_gtsm_stations.py` (required, section 1);
- `tools/list_stale_figures.py`;
- `tools/make_data_table.py` (regenerates `docs/data.md`).

The `analysis/phase1_*.py` scripts are base-model diagnostics; you don't need them.

---

## 5. Output layout under `<results_dir>/{basin_id}/`

Every figure now goes to `figures/`, or to `figures/analysis/` when `analysis=True`. File names follow `NN_<stage>_<what>__<run label>`; for adaptation runs the run label is `{scenario}__{pre|post}_{strategy}`.

| old | new |
|---|---|
| `preprocessing_inputs/visuals/*.png` | `figures/NN_*.png` |
| `visuals/10_calibration/*` | `figures/11_calib_*` |
| `spin_up/01_inundation_ratio.png`, `02_flood_animation.mp4` | `figures/13_spinup_*` |
| `runs/{sc}/visuals/02_flood_animation.mp4` | `figures/50_run_flood_progression__{sc}.mp4` (only with `analysis=True`) |
| `runs/{sc}/visuals/01_inundation_ratio.png` | `figures/analysis/15_run_inundation_ratio__{sc}.png` |
| `runs/{sc}/attribution_mask.png` | `figures/61_adapt_attribution_mask__{sc}.png` |
| `runs/{sc}/spinup_max_flood_depth.tif` | `runs/{sc}/calibration_max_flood_depth.tif` (see below) |
| `adaptation/{pre,post}/{st}/01_inundation_ratio.png`, `02_flood_animation.mp4` | `figures/62_adapt_inundation_ratio__{sc}__{pre,post}_{st}.png`, `figures/63_adapt_flood_progression__...mp4` |
| `{basin}_sea_mask.tif` | removed (the adapted skeleton keeps its own) |

**Unchanged paths:**
- `runs/{sc}/visuals/max_flood_depth.tif`;
- `runs/{sc}/visuals/flood_timeseries.csv` (now written only with `analysis=True`);
- `attribution_mask.tif`, `baseline_excess_volume.json`, `runs/metrics_comparison.csv`;
- the adaptation `flood_metrics.csv`, `max_flood_depth.tif`, `sfincs/` and `sfincs_skeleton/`.

**Please review: attribution class 4.** Class 4 is now every pixel wet in rule 11's last crest-calibration round, the embanked system at its protection standard. Before, it was every pixel wet in the spin-up.
- This lowers `baseline_excess_volume`, which water retention is sized on, and changes which pixels the post-processing measures clear.
- Ebro: class 4 grows from 15.35 to 20.59 km², and the flood volume outside it falls by 3.9 %.
- This change is uncommitted on the branch. See the top entry of `CHANGELOG.md`.

---

## 6. Where to read more

- `README.md`: overview and quick start.
- `docs/install.md`: env, site settings, pre-commit.
- `docs/data.md`: dataset table with how to get each one; "Adaptation geometries" lists the files per strategy.
- `docs/running.md`: targets, `raw_io`, adaptation runs.
- `docs/methods.md`: how each stage works.
- `CHANGELOG.md`: what changed and why, newest first.
