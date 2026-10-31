# Read-out: river depth, GVF and calibration (state before the rework)

Read-only survey of the repository and of `<results_dir>` (D:/GCFM_UU/results) and `<raw_data_root>`
(D:/GCFM_UU/raw_data). Code locations are `file:function` with line numbers at commit `edaf0be`; paths
are relative to the repository unless stated.

## A. Data in the catalogue

### SWOT
- **No SWOT product is catalogued or on disk**: no RiverSP reach, no RiverSP node, no L2 HR, no SWOT folder under `<raw_data_root>`.
- **SWOT statistics inside SWORD v17c** (catalogue entry `river_network`, `config/data_catalogue.yml:95-165`). These are the only SWOT data available:
  - reaches: `wse_obs_p10..p90`, `_range`, `_mad`, `width_obs_*`, `slope_obs_*`, `slope_obs_reliable`, `slope_obs_quality`, `slope_obs_n_passes`, `n_obs`, `swot_obs`, `swot_obs_source`;
  - nodes: `wse_obs_p10..p90`, `_range`, `_mad`, `width_obs_*`, `n_obs`.
- **Node-level WSE exists only as percentiles over all overpasses**: there is no per-pass time series. Slope exists at reach level only.
- **Cycles / time span**: not stated in `SWORD/v17c_release_notes.md` or the variable reference. Slopes there come from pass-level regressions of RiverSP node `wse_sm`, and `slope_obs_p50` is filled on 173,732 reaches.
- **Vertical datum of `wse_obs`**: not stated. The static SWORD `wse` is on MERIT/EGM2008.
- **How the pipeline uses SWOT**: only `slope_obs_p50` and `slope_obs_reliable`, as a check (the surface/SWOT slope ratio):
  - `workflow/src/water_surface.py:swot_check`;
  - `analysis/check_river_bed.py:swot_ratios`.
- `wse_obs_*` is not used anywhere. `src/sword.py:read_sword_nodes` reads only `reach_id`, `x`, `y`, `width`.
- **EGM2008 → GOCO06s conversion**: `workflow/src/raster.py:compute_geoid_offset_arr` (pyshtools). It is applied only to FathomDEM, in `workflow/scripts/06_get_elevation.py` (rule `get_elevation`). GEBCO and COAST-RP get the MDT instead (`src/surge.py:load_mdt`, `apply_mdt_correction`).
- **The datum conversion is not applied to SWOT data.** Only slopes are used, and those are datum-free.

### SWORD
- **Version**: v17c, files `SWORD/v17c/sword_{AF,AS,EU,NA,OC,SA}_v17c.gpkg`, layers `reaches` and `nodes`. `SWORD_global_v17c_unpublished(_modified).gpkg` is also on disk but unused.
- **IDs**: `reach_id` has 11 digits and `node_id` 14 digits, the standard SWORD format that RiverSP keys on.
  - The release notes say node x/y were restored to v17b "for SWOT D0-D2 continuity".
  - **Whether the IDs match a SWOT product is not verified**, because no SWOT product is on disk.

### GTSM / COAST-RP
- **COAST-RP** (`CoastRP/COAST-RP.nc`, plus `_ETC` and `_TC`): storm-tide **peak levels** only, at 23,226 GTSM stations, RP 1–1000 (10 levels).
- **GTSMv3 hourly reanalysis 1950–2024**: total water level (`gtsm_total_water_level`) and surge residual (`gtsm_surge_residual`).
  - Subset `GTSM_storm_tide_hourly/station_subset.nc`: 75 stations × 657,456 h for the 9 basins (`tools/subset_gtsm_stations.py`).
  - Rule `extract_gtsm_series` writes the subset per basin to `gtsm_hourly.nc` (e.g. 8 stations for 3279946).
- **Hydrographs are built.** Rule `storm_tide_hydrographs` (`workflow/rules/07a_storm_tide_hydrographs.smk`, `src/storm_tide_hydrograph.py`, HGRAPHER method) writes `storm_tide_hydrographs.nc`:
  - `average_tide` per station;
  - `surge_shape` per station and RP (10 RPs), ±60 h.
- Rule 07d copies these into `surge_forcing.nc` as `hg_tide_m` and `hg_surge_shape`. `src/surge.py:build_design_surge_matrix` (`storm_tide_event`) builds average tide + scaled surge peaking at the COAST-RP level. **This hydrograph drives the main runs only** (rule 14).
- **Not catalogued**: `GTSM_storm_tide_rp_hg.nc` (323 stations, POT-GPD return levels with hydrographs; made by `tests/KL_gtsm_storm_tide.py`, mentioned only in CHANGELOG.md).
- **Calibration (rule 11) does not use the hydrograph.** It holds a constant level from t = 0 (`workflow/scripts/11_depth_estimation_modelled.py`, lines 1033–1054 and 1209–1229):
  - the level is `storm_tide_at_rp_interpolated(surge_ds, surge_protection_rp_yr)`: the COAST-RP peak, interpolated linearly in RP at the FLOPROS coastal RP, plus MDT, rounded to 1 cm, per station;
  - without a FLOPROS coastal RP, it is `calm_sea_level_m`.

### GloFAS
- **Daily series per basin** (checked on 3279946):
  - `glofas_clip.nc` (`dis24`): 1979-01-02 to 2026-01-01, 17,167 days;
  - `river_forcing.nc` `discharge_daily`: same span, per crossing (bias-corrected where GRDC correction applies; it is disabled, `config/config.yml:220`).
- **No stored peaks or events table exists.** `river_forcing.nc` keeps only summary statistics (`pot_threshold`, `pot_r_days`, `pot_peaks_per_year`, `gpd_*`, `gev_shape`, `discharge_rp_table`), from `src/extreme_values.py:analyse_cell`.
- Events are picked on the fly in `src/river_forcing.py:real_flood_event_matrix` (`event_hydrograph.source: glofas_event`): independent peaks at least 14 days apart, choosing the one nearest the return level.

### Truth data (Rhine-Meuse 1248635, Mississippi 4267691, Ebro 2433835)
- **None exist**, in the catalogue or on disk: no surveyed bathymetry, no levee or dike inventory, no national flood map. Bathymetry is global only (GEBCO 2026). Protection data is FLOPROS design return periods only.
- **Other validation data:**
  - GRDC: `GRDC/GRDC-Daily.nc`, 444 stations; used for the (disabled) GloFAS bias correction and by `analysis/inspect_grdc_correlation.py`.
  - GESLA v4.1 (`raw_data/GESLA_v41/`): not catalogued; used only by `analysis/inspect_gesla_gauges.py`.

## B. Rule 10 (surface) and rule 11 (bed)

### Where each step happens

| step | location |
|---|---|
| node generation | `src/gvf_target.py:build_node_network` (l.214); called by `src/water_surface.py:build_water_surface` (l.375), rebuilt identically in `src/river_bed.py:estimate_river_bed` (l.265); `surface_on_network` (l.217) checks ids and positions (≤ 1 m) |
| node median | `water_surface.py:sample_centrelines` (l.114, land-use water mask) → `_split_samples` (l.219) → `node_medians` (l.152, ±`median_half_window_m`, stops at structures); empty nodes from neighbours, still-unset ones to sea (`build_water_surface` l.410-416) |
| sea pin | `water_surface.py:sea_mouths` (l.280); `build_water_surface` l.421-441 (`target[mouths] = sea`, `target = max(target, sea)`), re-floored after smoothing (l.448) |
| isotonic fit | `gvf_target.py:enforce_non_increasing` (l.457, weighted Hildreth dual ascent + monotone clip l.507-516); called at `build_water_surface` l.443 and after each smoothing (l.447) |
| 5 km / 40 km smoothing | `build_water_surface` inner `smoothed()` (l.445-448) → `gvf_target.smooth_on_network` (l.402, repeated neighbour averaging, n_pass = W²/(6·spacing²)); level with `smoothing_m` (l.450), slope surface with `slope_smoothing_m` (l.451) |
| slope | `gvf_target.py:main_path_slope` (l.533; largest-Q path, Q = `bankfull_discharge_acc_m3s` via `water_surface._reach_discharge` l.265), called at `build_water_surface` l.452 |
| Manning inversion | `river_bed.py:manning_depth` (l.203; width ≥ 1 m) → `gvf_solver.normal_depth` (brentq, l.50-61); called in `estimate_river_bed` l.297 |
| slope floor | `estimate_river_bed` l.285-286 (`depth_slope = max(slope, slope_floor)`); flag `RiverBed.on_floor` (l.101) |
| width smoothing | `river_bed.py:node_widths` (l.160; SWORD node width scaled to the reach width, running median ±W/2, then `smooth_on_network`); called in `estimate_river_bed` l.293 |
| tiny-flow fallback | `estimate_river_bed` l.298-308: where q ≤ `q_min_m3s`, h = c·W^b with `gvf_solver.fit_width_depth` (l.151; b clipped to [0, 1]; fewer than 3 points → c = median, b = 0) |
| GEBCO handover | `estimate_river_bed` l.311-330: nodes of class `water_body` with finite DEM below `gebco_shallow − 0.1` take the DEM (GEBCO) bed; linear blend within `bathymetry_handover_m`; `gebco_shallow_m = sea − gebco_min_depth_m` (`estimate_river_bed_from_files` l.459); classes from `gvf_target.classify_nodes` (l.581) |
| forward check | `estimate_river_bed` l.336-341: `gvf_solver.forward_network` at the design Q → `forward_level_m`; the same at the mean Q → `mean_level_m` (l.342-353); summaries `forward_deviation_summary` (l.473), `_log_summary` (l.490) |
| burn | rule 10 does not burn: it sets channels to the surface (`water_surface.py:condition_channels_to_surface`, l.579). Rule 11: bed at the centreline cells (`11_depth_estimation_modelled.py` l.894, anchors l.950) → `src/river_burn.py:burn_river_channel` (l.46, subgrid pixel grid; call l.959-969) → `constrain_to_coarse_channel_mask` (l.971) → `mean_elevation_on_coarse_grid` (l.1007) → written into `dep` (l.1014) |

### Constants

Keys are relative to `river_processing` in `config/config.yml` unless stated otherwise.

| constant | key | value |
|---|---|---|
| Manning n | `channel_manning_n` | 0.03 |
| node spacing | `water_surface.node_spacing_m` | 500 m |
| node median window | `water_surface.median_half_window_m` | 1000 m |
| level smoothing | `water_surface.smoothing_m` | 5000 m |
| slope smoothing | `water_surface.slope_smoothing_m` | 40000 m |
| mouth tolerance | `water_surface.mouth_level_tol_m` | 0.5 m |
| water-body class | `water_surface.water_body` | `{min_width_m: 1000, sea_class_fraction: 0.5}` |
| structure step | `water_surface.step_threshold_m`, `structure_min_drop_m` | 0.5 m, 0.2 m |
| design discharge | `river_depth_modelling.bed.design_discharge` | `bankfull` |
| slope floor | `river_depth_modelling.bed.slope_floor` | 1.0e-5 |
| width smoothing | `river_depth_modelling.bed.width_smoothing_m` | 5000 m |
| GEBCO handover | `river_depth_modelling.bed.bathymetry_handover_m` | 5000 m |
| tiny-flow threshold | `river_depth_modelling.bed.q_min_m3s` | 1.0 m³/s |
| GEBCO clamps | `terrain.gebco_min_depth_m`, `terrain.gebco_max_depth_m` | 2.0 m, 10.0 m |
| Froude cap | **not in config** | 0.95 (hard-coded, below) |
| ramp | `river_depth_modelling.ramp_hr` | 6.0 h |
| run length | `river_depth_modelling.calibration_margin_hr`, `calibration_max_days` | 12 h, 4 days |
| regime thresholds | `river_depth_modelling.wave_regime_rT` | [0.3, 3.0] (inertial / transition / frictional) |
| convergence check | `river_depth_modelling.steady_check` | `{window_hr: 3, tol_m: 0.05}` |
| confinement weir crest | `river_depth_modelling.weir_crest_m` | 1000.0 m |
| round 2 | `river_depth_modelling.verify_crests` | false |
| boundary check | `testing.upstream_boundary_check` (top-level section, not under `river_processing`) | `amplitude_threshold_fraction` 0.05, `level_tol_m` 0.05, `margin_km` 5, `stress_surge_rp_yr` 100, `stress_slr_m` 1.0, `high_gradient_elevation_m` 10 |

**Hard-coded, not in config:**

- **GVF solver** (`src/gvf_solver.py`):
  - `froude_cap = 0.95`: default of `integrate_reach` (l.81) and `forward_network` (l.173). `river_bed.py` l.341 and l.343 do not pass it.
  - `max_step_m = 50` (l.82).
  - `h_dry_m = 0.01` (l.83).
  - normal-depth bracket 1e-4 to 1e5 m (l.56-59), slope minimum 1e-12 (l.54), `xtol = 1e-6` (l.61).
  - fit fallback c = 1.0 (l.158).
- **River profile** (`src/river_profile.py`):
  - `forward_levels` `froude_cap = 0.95` (l.93).
  - `walled_travel_time` `min_depth_m = 0.1` (l.139); `regime_rT` default (0.3, 3.0) (l.138), which config overrides.
  - kinematic factor 5/3 (l.179), g = 9.81 (l.6).
- **River bed** (`src/river_bed.py`):
  - width ≥ 1 m in `manning_depth` (l.207).
  - bathymetry margin 0.1 m (l.314).
  - forward-check bands (0.5, 1.0) m (l.68).
  - node offset tolerance 1.0 m (l.233).
- **Node network** (`src/gvf_target.py`):
  - isotonic `tol_m = 1e-3`, `max_sweeps = 20000` (l.460-461).
  - lift warning 0.05 m (l.515).
  - GEBCO-floor tolerance 0.1 m in `classify_nodes` (l.600).
- **Water surface** (`src/water_surface.py`):
  - `SWOT_RATIO_SPAN = 2.0` (l.88).
  - 5×5 node sampling disc (l.497).
  - `LANDUSE_SEA_CODE = 200`, duplicated from `src.landuse` (l.73).
- **River burn** (`src/river_burn.py`):
  - `burn_river_channel` `margin_m = 500` (l.53).
  - coarse-mask fraction > 0.5 (l.340).

The 5000 m and 40000 m smoothing lengths come from config only.

### `design_discharge`

**Read** in `src/river_bed.py:estimate_river_bed_from_files`, l.411-415. `DESIGN_DISCHARGE_VAR` (l.377) maps it to `bankfull_discharge_m3s` or `mean_discharge_m3s` in `river_forcing.nc`. It is also a required key at l.62.

**What depends on its value.** The seed Q (l.429) is routed by `accumulate_discharge` to `q_design` (l.436). From there:

- **In `estimate_river_bed`:**
  - Q per node (l.296), the Manning depth (l.297) and the tiny-flow fit (l.298-308);
  - the bed (l.310) and the handover blend;
  - `main_path` (l.332-335);
  - `forward_level_m` (l.341);
  - `mean_level_m`: its Q is the mean, but it is solved on the design bed.
- **In rule 11** (`11_depth_estimation_modelled.py`):
  - `rivers.q_design_m3s` (l.885);
  - centreline bed and depth (l.894-906);
  - the node table (l.915-921);
  - reach columns `water_slope`, `water_level_m`, `mean_flow_depth_m`, `depth_on_slope_floor` (l.922-929);
  - round-0 `calibration_state.csv` (l.934) and `river_depth_segments.gpkg` (l.943);
  - the burn (l.950-1014), and so the main-grid `dep` and the subgrid table reused by rule 12a;
  - zsini river cells from `mean_level_m` (l.1056-1069; rule 12a reads the same column, `12a_build_sfincs_skeleton.py` l.333);
  - the walled travel time and run length (l.1131-1164): Q is the calibration Q, the bed is the design bed;
  - the water levels of rounds 1–2, and therefore the weir crests and `coastal_protection_weir.gpkg`;
  - the outputs `river_network_depth_estimated.gpkg` and `river_burned_*.tif`.
- **Downstream readers of the node table:** `river_profile.mainstem_profile` (l.74) and `scaled_discharge` (l.84); `analysis/_upstream_profile.py`, `analysis/estimate_peak_travel.py` (l.241), `analysis/benchmark_peak_travel.py`.
- **Others:**
  - `analysis/check_river_bed.py` l.423-436 chooses the bankfull or mean Q from it;
  - `rules/19_analysis_figures.smk` l.109 passes it to `scripts/19b_analysis_calibration.py` as a figure label only.
- **Not affected:** the rule-10 slope always uses the routed bankfull Q.

### Forward-check residuals
- **Node table**, `<results_dir>/{b}/preprocessing_inputs/visuals/11_calibration/river_bed/{b}_river_bed_nodes.csv` (CSV, plus a `.gpkg` of points in the model CRS), written at rule-11 l.913-921.
  - Columns (`RiverBed.node_table`, `river_bed.py` l.105): node, reach_id, x, y, dist_to_sea_m, fathom_level_m, informative, slope, depth_slope, on_floor, width_m, q_m3s, depth_m, level_m, bed_m, bathymetry, main_path, forward_level_m, mean_level_m.
  - **There is no residual column**: the residual is `forward_level_m − level_m`.
- **Per node pair**, `<results_dir>/{b}/preprocessing_inputs/diagnostics/{b}_river_depth_segments.gpkg` (GeoPackage, one line per consecutive node pair; `river_bed.py:depth_segments` l.531).
  - Columns: reach_id, depth, width, `forward_minus_surface_m` (pair mean of forward − level), fallback (`on_floor`), geometry.
  - This file was added at `edaf0be`, and **no basin has it on disk yet**.
- **Summary statistics are logged only** (`_log_summary`), not written to a file.

### Bank-top or crest along the centreline
- **No bank-top elevation is extracted anywhere.**
- The crest/DEM/bed profile (`figures/30_calib_crest_profile.png`, `src/plots.py:plot_crest_profiles`) is drawn from:
  - `round{i}/calibration_state.csv`, written by rule-11 `_write_calibration_state` (l.733);
  - read by `src/river_depth_calibration.py:gather_calibration_round_profile` (l.71);
  - a copy goes to `visuals/11_calibration/round_profiles_seed{seed}.csv`.
- **Columns:** the centreline cells (`reach_id`, `row`, `col`, `along_m`, `x`, `y`; from `river_burn.build_centerline_cells_regular`), plus `dem_m`, `river_depth_m`, `weir_crest_m`, `zs_m`.
- **What `dem_m` is:** the pre-excavation conditioned SFINCS-grid elevation of the centreline cell (`grid.bed_elevation_m`, rule-11 l.427).
  - Rule 10 set channel pixels to the smoothed water surface, so `dem_m` is about the water surface where the channel fills the cell.
  - It is **neither the bank top nor the natural DEM.**
- `river_depth_m = max(dem_m − bed, 0)` (l.906), so the plotted bed is clipped at `dem_m`.
- `weir_crest_m` is the production crest of the last round. It is the edge crest mapped to the centreline cell, not a bank elevation.

## C. GVF solver and routing

### Network solver
- **Entry point**: `src/gvf_solver.py:forward_network` (l.166-219). It integrates node-to-node segments with `integrate_reach` (l.73-148).
- **Inputs:**
  - `net`: a `GvfNetwork` from `gvf_target.build_node_network`;
  - `bed_m[node]` and `width_m[node]` (a segment's width is the mean of its two nodes);
  - `reach_q`: one Q per reach;
  - scalar Manning `n`;
  - `end_levels`: the level at each downstream network end;
  - `froude_cap` (default 0.95).
- **Output**: the level per node only. No Froude number and no flags are returned.
- **Algorithm:**
  - reaches are processed downstream first (`reversed(net.reach_ids)`), integrating upstream;
  - the steady GVF ODE dh/ds = −(S0 − Sf)/(1 − Fr²) is solved with RK4 in substeps of ≤ 50 m: a direct ODE integration, not a standard step. The section is rectangular, with Manning Sf and R = A/(w + 2h);
  - depth is floored at `max(critical_depth(q, w, froude_cap), h_dry_m)`, and the denominator at 1 − cap²;
  - a reach with Q ≤ 0 is flat at its downstream level, and the downstream level is clipped to at least the bed (l.199-202);
  - at a node shared by several reaches, a flowing reach's level replaces a dry one's (l.216-218).

### Routing and bifurcation split
- **Routing and the width-proportional split** are both in `src/river_network.py:accumulate_discharge` (l.386-474):
  - one topological pass sums flows at confluences;
  - at a split, `shares = w_dn / total_w` (l.448-458); arms split equally when all widths are 0.
- **Split fractions are not a solver input.** The solver receives only the already-routed per-reach Q.
- **Canal arms:**
  - excluded by `river_network.py:canal_excluded_arms` (l.319-383), configured by `river_processing.canal_routing` (`auto` plus force lists);
  - computed in `scripts/08_clean_river_network.py` l.157-170 and stored as column `zero_discharge_share`;
  - read back as `excluded_arms` in `river_bed.py:estimate_river_bed_from_files` (l.416-424), rule-11 l.238-248 and `analysis/_upstream_profile.py:routed_discharge` (l.168-184);
  - excluded arms get width 0; if every arm is excluded, they keep their share (l.449-454).

### Junction level and Froude cap
- **Junction downstream level** (`forward_network` l.188-191): the **unweighted arithmetic mean** of the upstream-end levels of the flowing downstream arms. If no arm flows, all arms are used. It is not the min, the max or a Q-weighted mean.
- **Network ends** take `end_levels`. In `estimate_river_bed` (l.336-340) that end level is rule 10's surface level at the end node, not the sea level.
- **Froude cap**: applied in `integrate_reach` (l.103, l.113-115, l.144-145), whenever the depth falls below the depth at which Fr equals the cap.
  - The flag is `ReachSolve.capped`, a boolean per node (l.70, l.146).
  - `forward_network` drops it, so the network result carries no cap flag.

### Callers
- **Network solver (`forward_network`):**
  - `river_bed.py:estimate_river_bed` l.341: `forward_level_m` at the design Q.
  - `river_bed.py:estimate_river_bed` l.342-353: `mean_level_m` at the routed `mean_discharge_m3s`.
  - zsini river level: stored `mean_level_m`, used via `water_surface.py:river_cells_at_surface` (l.662) in rule 11 (l.1057-1067) and rule 12a (l.333-339). The solver is not called again.
  - `analysis/check_upstream_boundary.py:check_basin` l.228: Q_mean and Q_bf crossed with calm and extreme sea levels.
  - `analysis/benchmark_upstream_boundary.py` l.229: 0.5·Q_mean, Q_mean, Q_bf.
  - tests: `tests/unit/test_gvf.py`, `tests/unit/test_upstream_boundary.py`.
- **Single-path model** (`src/river_profile.py:forward_levels`, l.88): calls `integrate_reach` along one main stem, with the sea level as its downstream boundary.
  - `river_profile.py:walled_travel_time` (l.132): the calibration run length in rule 11 (l.1143-1152). Q is the calibration Q, the sea level is the highest calibration station level, and the slowest crossing sets the run length.
  - `analysis/benchmark_peak_travel.py` l.153-160: the calibration Q.
  - `analysis/_upstream_profile.py:surface_mismatch` (l.46), `analysis/check_river_bed.py` (l.437, l.442) and `analysis/estimate_peak_travel.py` (`river_leg`, l.74): these use the design Q, the mean Q or the event-peak Q.
  - tests: `tests/unit/test_upstream_profile.py`, `tests/unit/test_river_profile.py`.
- **Stored `forward_level_m` without a solver call:** `river_bed.py` l.475 and l.535.

## D. Calibration and SFINCS setup

### Rules
- **Calibration, rounds 0–2: one rule**, `modelled_depth_estimation` (`workflow/rules/11_depth_estimation_modelled.smk`, script `11_depth_estimation_modelled.py`).
  - Inputs: conditioned elevation, clean river network, `river_forcing.nc`, `surge_forcing.nc`, `protection_levels.json`, land use and roughness on the grid, `sfincs_grid.json`.
  - Production outputs: `*_river_network_depth_estimated.gpkg`, `*_river_burned_subgrid.tif`, `*_river_burned_dem_sfincs_grid.tif`, `*_coastal_protection_weir.gpkg`, `*_zsini_sea_cells_on_grid.tif`, and the round-1 subgrid table and mask (`calibration_subgrid_files`).
  - Per-round outputs (`calibration_round_figures`, `00_common.smk`): `round{r}/sfincs_map.nc`, forcing and max-depth figures, water level CSV; diagnostics in `preprocessing_inputs/diagnostics/`.
- **Round 1** (confined, l.1253–1332) and **round 2** (production weir, l.1335–1380) are sections of the same script.
  - Round 2 runs only with `verify_crests: true`, which is currently false. It only warns about overtopped edges and changes nothing.
  - With round 2 off, the script deletes `round2/` (l.1331).
- **Spin-up**: `run_spinup` (`13a_run_spinup.smk`).
  - Inputs: skeleton `sfincs.inp`, `land_bed_level.nc`, surge and river forcing.
  - Outputs: restart file, `sfincs_map.nc`, `sfincs_his.nc`.
  - Forcing: 2 days of constant mean discharge; the sea ramps to its calm level over 6 h.
- **Main runs:**
  - `build_sfincs_skeleton` (12a) reuses the round-1 subgrid table and imports the rule-11 weir unchanged;
  - `build_sfincs` (14) writes `runs/{scenario}/sfincs/sfincs.inp` and forcing;
  - `run_event` (15) writes `sfincs_map.nc`.
- **Consumers of round-2 outputs:**
  - no production rule (12a, 13a, 14, 15);
  - `analysis_calibration` (rule 19) and the Snakefile targets, only when round 2 is enabled (`CALIB_ROUNDS`);
  - `analysis/check_crest_calibration.py`, which hard-codes `round2/`;
  - `src/river_depth_calibration.py` (l.105), which reads `round{i}/calibration_state.csv` for each round present.

### Calibration sea boundary and storm-tide floor
- **Sea boundary**: each station is held at the FLOPROS-RP storm-tide level **from t = 0, with no ramp and no hydrograph** (l.1039–1045, boundary at l.1208–1222, `bnd_dist` 5000 m).
  - Only the rivers ramp: a half-cosine over `ramp_hr` (6 h) from Q_mean to the protection-RP Q (l.1182–1187).
- **Floor in zsini**: `calib_sea_level_m = coastal_design_peaks.max()` (l.1054). Then `raise_wet_cells_to(zsini, sea_wet | channel_mask, calib_sea_level_m)` (l.1073–1075), applied after the river cells are set to the mean-discharge level (l.1067).
- **Floor on crests**: `crest = ceil_cm(max(water-side zs_max, coastal_protection_crest_m) + freeboard)` in `src/protection_weir.py:build_coastal_protection_weir` (l.665–668), called at rule-11 l.1265.
  - In addition, crests are raised to the face's subgrid `uv_zmin` + 1 cm (`floor_crests_above_face_zmin`, rule-11 l.1280–1299; `protection_weir.py` l.285).

### Run length and convergence check
- **Run length**: `run_hr = ceil(ramp_hr + T_walled + calibration_margin_hr)` at rule-11 l.1164.
  - T_walled is the largest last-node `t_wave_hr` over all crossings, from `walled_travel_time` (l.1136–1163).
  - Cap: if `run_hr > calibration_max_days · 24` (96 h), the rule raises ValueError, a hard failure (l.1171).
- **Convergence check**: `_steady_check` (l.1235–1250), called after round 1 (l.1261) and round 2 (l.1348).
  - It counts centreline cells whose zs rose more than `tol_m` over the last `window_hr` and logs `"[round] N/M centreline cell(s) rose more than 0.05 m over the run's last 3 h (max …) -- not steady at the run end: raise calibration_margin_hr"`.
  - **It warns only; it never fails.**

### SFINCS
- **Version**: SFINCS v2.3.0 "mt. Faber Release", build 2025-11-18 (`sfincs.log`), executable `SFINCS_v2.3.0_mt_Faber_release_exe/sfincs.exe` (`config/site.yml`).
- **Mesh**: regular in every run (log: "using regular mesh", "Quadtree refinement: no").
  - The binary (v2.x) and the installed hydromt_sfincs (`components/quadtree/*`) support quadtree; the repository does not use it.
- **Subgrid** (`config.yml:283–285`): `nr_subgrid_pixels` 10, `nr_levels` 20, `nrmax` 2000.
- **sfincs.inp settings** (hydromt_sfincs defaults; none is set in repository code):
  - `huthresh` 0.01, `alpha` 0.5;
  - `advection` 1 (first-order upwind);
  - `viscosity` 1, `nuvisc` 0.01 (the log shows 1.0 after grid scaling);
  - `btfilter` 60, `cdnrb` 3;
  - Manning from `manningfile`;
  - "mean depth in subgrid cell" for hmax.
- **Not set**: no uvmax/uvlim, no conveyance or `subgrid_q` option, no theta.

### Weirs and structures
- **`sfincs.weir` format**: ASCII, one block per grid-edge segment: `weirNN`, then `2 4`, then two lines `x y z par1`.
  - Formats: x and y `%11.1f`, z `%.2f`; par1 = Cd 0.60.
  - Writer: `src/sfincs_run.py:write_weir_file` (l.230–255). Reader: `read_weir_file` (l.188).
  - 3279946 has about 29,950 segments.
- **The "1000 m wall" is a crest height (`weir_crest_m`), not a buffer width.** The confinement weir is `build_coastal_protection_weir(..., weir_crest_m, water_side_crest_on_grid = all-NaN, channel_mask_gap_free = True)` (rule-11 l.813–823).
  - Channel mask: `src/river_burn.py:build_channel_mask_regular` (l.371), reach centrelines buffered by width/2 (`_flush_capped_buffer`, l.490–499) and rasterized with `all_touched`.
  - `water_like = ocean | channel_mask` (`protection_weir.py` l.577). After small-component removal, the walls are the land / water_like cell edges (`_seaward_edges_regular`, l.178–249).
- **Crossing check**: no explicit check that a weir line crosses faces inside the channel buffer.
  - By construction, edges sit only between land and water_like cells, so no wall sits between two channel cells.
  - The only face-level check is crest vs `uv_zmin` (`floor_crests_above_face_zmin`).

### Cross-sections and observation points
- **`crsfile`**: not configured in any run; no `crsfile` key exists in config, scripts or sfincs.inp. The "cross-section" code in rule 11 (l.461–578) is internal max-zs sampling of grid cells from `sfincs_map.nc`.
- **`obsfile`**: only in the skeleton (`12a_build_sfincs_skeleton.py` l.454–532), shared by the spin-up and event runs.
  - Placement: the 2 crossings with the highest bankfull Q (`n_top_crossings`), traced downstream along SWORD `rch_id_dn`, with `n_per_crossing` = 5 points along each traced line (`config.yml:286–289`). Some points fall in the sea.
- **Calibration rounds have no obsfile**: they read zs per cell from `sfincs_map.nc`.

## E. State on disk and cost

### Outputs per basin
The 9 basins come from `DeltaWebs/modified/9_polygons.gpkg` (`BasinID2`). `<results_dir>` has 8 basin folders; Mississippi has none.

| basin | delta | spin-up restart | round 1 map | round 2 map | main runs (`sfincs_map.nc`) | forward check (node table) | upstream boundary check |
|---|---|---|---|---|---|---|---|
| 3279946 | Chao Phraya | yes | yes | yes* | none (`runs/default` has inputs only) | yes | yes (+ benchmark) |
| 4267691 | Mississippi | no results folder | – | – | – | – | no |
| 1416812 | Zambezi | no | yes | yes* | none | yes | yes |
| 1475442 | Mangoky | no | yes | yes* | none | yes | yes |
| 2433835 | Ebro | no | yes | yes* | none | yes | yes (+ benchmark) |
| 1248635 | Rhine-Meuse | no | no | no | none | no | yes |
| 2444235 | Vjosë | no | yes | yes* | none | yes | yes |
| 2209166 | Taiwan | no | yes | yes* | none | yes | yes (+ benchmark) |
| 620947 | Calvert | no | yes | yes* | none | yes | yes |

\* Round 2 is left over from runs made with `verify_crests: true`; rule 11 deletes it at its next run.

- **Forward check**: the node tables hold `forward_level_m` and `level_m`. The `river_depth_segments.gpkg` diagnostics do not exist yet anywhere.
- **Upstream boundary check outputs** are in the repository's `figs/upstream_boundary_check/`, the old analysis location. None are under `figures/analysis/` yet.

### Wall-clock time
**Current outputs.** From the "Total time" line in each `sfincs.log`:

| basin | spin-up | round 1 (configured run length) | round 2 |
|---|---|---|---|
| 3279946 | 24.4 min | 37.5 min (59 h) | 36.5 min |
| 1416812 | no log | 15.5 min (41 h) | 15.7 min |
| 1475442 | no log | 0.73 min (24 h) | 0.65 min |
| 2209166 | no log | 0.47 min (20 h) | 0.49 min |
| 2433835 | no log | 0.26 min (20 h) | 0.29 min |
| 2444235 | no log | 0.09 min (19 h) | 0.09 min |
| 620947 | no log | 0.23 min (26 h) | 0.24 min |
| 1248635 | – | not run (fails before SFINCS) | – |
| 4267691 | – | ran (90 h run length, log of 27 Oct); its `sfincs.log` went with the results folder | – |

**Main runs and older pipeline (9–10 Oct).** There is no current main-run log. The timestamps in `logs/run_all_*.log` are indicative only (stale per TODO B30):

| basin | rule 11 (whole rule) | spin-up | event run |
|---|---|---|---|
| 1248635 | 211.7 min | 16.5 min | 255.6 min |
| 1475442 | 29.5 min | 1.8 min | 32.5 min |
| 2209166 | 24.6 min | 1.7 min | 24.0 min |
| 2433835 | 12.1 min | 0.9 min | 17.0 min |
| 2444235 | 2.2 min | 0.4 min | 4.9 min |
| 620947 | 14.2 min | 1.2 min | 16.2 min |

### Run-length cap and convergence failures

| basin | 4-day cap | steady check |
|---|---|---|
| 1248635 Rhine-Meuse | **fails**: walled travel times 131.3, 516.2, 33.2 and ∞ h, so run_hr = ∞ | not reached |
| 3279946 Chao Phraya | passes (59 h) | **fails, both rounds**: 9172 of 12462 cells rose > 0.05 m, max +0.77 m (TODO B34) |
| 1416812 Zambezi | passes (41 h) | **fails, both rounds**: 18 of 2720 cells, max +0.52 m |
| 4267691 Mississippi | passes (90 h, under the 96 h cap) | not evaluable: the round-1 period-max level is NaN; results missing |
| 1475442, 2209166, 2444235, 620947 | pass | pass (max rise +0.00 to +0.01 m) |
| 2433835 Ebro | passes | passes (0 of 835 cells, max +0.04 m) |

## F. Repository hygiene

### Tests
There are no notebooks in the repository. Tests touching these modules (69 in total):

| area | file | tests |
|---|---|---|
| rule 10 surface | `tests/unit/test_water_surface.py` | 15 |
| rule 11 bed | `tests/unit/test_river_bed.py` | 9 |
| burn | `tests/unit/test_river_burn_smoke.py` | 5 |
| GVF solver and node network | `tests/unit/test_gvf.py` | 13 |
| walled travel time | `tests/unit/test_river_profile.py` | 4 |
| upstream boundary | `tests/unit/test_upstream_boundary.py`, `tests/unit/test_upstream_profile.py` | 8 + 10 |
| peak travel | `tests/unit/test_peak_travel.py` | 5 |

The quick-look stage-10/11 panels are covered in `tests/unit/test_quicklook.py`.

**No tests for:** `src/river_depth_calibration.py`, `_steady_check`, the `calibration_max_days` failure, and the scripts `10_river_surface.py` and `11_depth_estimation_modelled.py`.

### Open items
There are no TODO / FIXME / XXX markers in the listed modules or the rule 10/11 scripts. Open rows:

| id | location | summary |
|---|---|---|
| B34 | `workflow/scripts/11_depth_estimation_modelled.py:991` | 3279946 not steady at 59 h, so crests too low; size the run on the settle time |
| B35 | `workflow/src/river_profile.py:132` | SFINCS sits 0.6–1 m above the 1D GVF (2209166: 1.5–3 m below); the run length rests on the 1D model |
| B33 | `workflow/src/river_bed.py:320` | GVF zsini vs old spin-up end on 3279946 (median −0.67 m); add the steady check to rule 13a |
| B26 | `workflow/scripts/11_depth_estimation_modelled.py:1092` | single unstable cells in the calibration rounds; accepted |
| B31 | `analysis/benchmark_upstream_boundary.py:1` | stress-case benchmark missing for 3279946; numbers still quoted at RP100 + 0.5 m |
| B32 | `analysis/check_event_timing.py:1` | no river-only run to validate the peak lag and walled travel estimate |
| B30 | `<results_dir>/{6 basins}/sfincs_skeleton` | skeletons stale vs current preprocessing |
| S2 | `workflow/src/river_bed.py:1` | validate the round-0 bed against surveyed bathymetry (no such data catalogued, see A) |
| S3 | `workflow/src/water_surface.py:73` | duplicated `LANDUSE_SEA_CODE` |
| HANDOFF | `HANDOFF_upstream_boundary.md:3-5` | open: Box 1 / Figure 1 and `L_surge` (section 11) |
| FWTT | `flood_wave_travel_time_methods.md:83-89` | `T_fill` exceeds 60 h after the surge peak in 1248635; B32 validation, overbank correction M5 |
| FWTT | `flood_wave_travel_time_methods.md:205-210` | run length = ramp + T_walled + 12 h; proposes the larger of T_adjust and the diffusive L²/D |
| FWTT | `flood_wave_travel_time_methods.md:337-346` | six open items (late-filling compartments, Ebro 168 h override, hazard-run `after` sizing) |
