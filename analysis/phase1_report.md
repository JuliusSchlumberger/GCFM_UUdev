# Phase 1 diagnostics: Ebro (2433835) and Chao Phraya (3279946)

Code at commit `5e4db0b` (river depth / GVF / calibration as in
`docs/readout_river_depth_gvf.md`). Every table names the script that
reproduces it (`python analysis/<script>.py [basin ...]`). Per-basin outputs
are in `<results_dir>/<basin>/figures/analysis/90_<script>_*`, and the
cross-basin summaries in `<results_dir>/figures/analysis/<script>_summary.csv`.

## 1.0 Baseline main run

Script: `phase1_run_times.py <snakemake log> <basin>`. Run: `snakemake all --cores 14 --config target_basins="[<b>]" target_scenarios="['default']"`.

| basin | jobs | wall-clock | rule 11 | spin-up | event run | flooded area | volume | failure |
|---|---|---|---|---|---|---|---|---|
| 2433835 | 17 / 17 | 15.8 min | 1.3 min | 1.2 min | 10.3 min | 270.2 km² (63.9 % of land) | 0.414 km³ | none |
| 3279946 | 13 / 17 | 63.7 min to the stop | 33.4 min | 22.4 min | not run | – | – | stopped after the spin-up by decision (see below) |

Other rules (minutes, Ebro / Chao Phraya):

| rule | Ebro (min) | Chao Phraya (min) |
|---|---|---|
| get_boundary_forcings | 1.38 | 1.53 |
| grid_align_landuse | 0.13 | 1.47 |
| build_sfincs_skeleton | 0.20 | 2.13 |
| get_elevation | 0.35 | 0.95 |
| land_bed_level | 0.12 | 0.87 |
| every other rule | under 0.6 | under 0.6 |

The rebuilt skeletons (TODO B30) and the rule 01–13a outputs of both basins are current.

## 1.1 Bankfull freeboard

Script: `phase1_freeboard.py`.

- **Bank top:** P90 of rule 06's merged DEM (FathomDEM, GOCO06s) in a band from W/2 to W/2 + 200 m either side of the centreline, ±250 m along it. The lower bank is used.
- **Nodes:** `water_body` nodes and nodes without bankfull flow are skipped.
- **Freeboard:** bank top minus the GVF level at Q_bankfull.
- **Bed options:**
  - (a) the stored bed;
  - (b) a mean-Q bed (`design_discharge: mean`, re-estimated with `estimate_river_bed_from_files`), solved at Q_bankfull with `forward_network`.
- **Reproducibility:** the re-estimate reproduces the stored `forward_level_m` exactly (max |Δ| 0.0000 m).

| basin | option | nodes | median | p5 | p95 | share < 0 m | share > 2 m |
|---|---|---|---|---|---|---|---|
| 2433835 | a bankfull bed | 136 | 3.46 m | −0.05 m | 28.96 m | 6.6 % | 66.9 % |
| 2433835 | b mean-Q bed | 136 | 1.69 m | −0.39 m | 27.14 m | 16.2 % | 45.6 % |
| 3279946 | a bankfull bed | 1987 | 1.18 m | −0.18 m | 4.54 m | 8.5 % | 22.9 % |
| 3279946 | b mean-Q bed | 1987 | −2.86 m | −5.67 m | 1.68 m | 87.5 % | 4.3 % |

- **Figures:** `90_phase1_freeboard_profile_{a,b}.png`, one line per reach on the largest-Q paths.
- **Ebro p95:** set by the valley walls in the upper 8 km.
- **Chao Phraya, mean-Q bed:** at bankfull the river overtops the banks almost everywhere.

## 1.2 Slope-floor census

Script: `phase1_slope_floor.py`.

- The bed is re-estimated at each floor.
- Depth changes are relative to 1e-5, over the flowing nodes on the floor at either value (the only nodes that change). The median over all nodes does not move, because fewer than half sit on the floor.

| basin | stored share on floor | floor | share on floor | affected nodes | median depth change |
|---|---|---|---|---|---|
| 2433835 | 28.7 % | 5e-6 | 24.3 % | 39 | +2.29 m (+24.0 %) |
| 2433835 | | 1e-5 | 28.7 % | – | reference |
| 2433835 | | 2e-5 | 34.6 % | 47 | −1.80 m (−19.2 %) |
| 3279946 | 25.3 % | 5e-6 | 21.5 % | 503 | +3.05 m (+24.3 %) |
| 3279946 | | 1e-5 | 25.3 % | – | reference |
| 3279946 | | 2e-5 | 30.4 % | 605 | −2.66 m (−19.5 %) |

## 1.3 Forward-check gate

Script: `phase1_forward_check.py`.

- Residual = `forward_level_m − level_m`, on flowing, non-bathymetry nodes.
- Gate: median within ±0.25 m and p95 of |residual| within 0.75 m.
- Written to `<results_dir>/<basin>/preprocessing_inputs/diagnostics/<basin>_forward_check.csv`.

| basin | nodes | median | p95 \|res\| | pass |
|---|---|---|---|---|
| 2433835 | 136 | +0.20 m | 0.88 m | **no** (p95) |
| 3279946 | 1987 | +0.44 m | 1.12 m | **no** (median and p95) |

## 1.4 Froude-cap flags

Script: `phase1_froude_flags.py`.

- Uses the stored bed solve, re-run with `forward_network(..., return_capped=True)`. Its levels equal `forward_level_m`.
- Capped reaches are listed in `90_phase1_froude_flags_reaches.csv`.

| basin | nodes | capped | reaches (capped nodes) |
|---|---|---|---|
| 2433835 | 136 | 0 | – |
| 3279946 | 1987 | 1 (0.05 %) | 44306300565 (1 of 21) |

## 1.5 SFINCS bifurcation split

Script: `phase1_bifurcation_split.py run|compare`.

- **Run:** round 1 rerun in `<results_dir>/../scratch_phase1/3279946/round1_crs/`, with cross-sections from `bifurcation_cross_sections` (the helper behind `river_depth_modelling.crosssections`). SFINCS took 26.7 min.
- **SFINCS share:** end-of-run |discharge| per arm over the arms of the bifurcation.
- **Width share:** `accumulate_discharge`'s share of the calibration Q.
- **Ebro:** no bifurcation on the forcing paths, so nothing to compare.

| basin | parent reach | arm | SWORD width | width share | SFINCS share | abs. error |
|---|---|---|---|---|---|---|
| 3279946 | 44306300031 | 44306300025 | 54 m | 0.162 | 0.184 | 0.021 |
| 3279946 | 44306300031 | 44306300125 | 279 m | 0.838 | 0.816 | 0.021 |
| 3279946 | 44306300125 | 44306300135 | 258 m | 0.804 | 0.775 | 0.028 |
| 3279946 | 44306300125 | 44306300565 | 63 m | 0.196 | 0.225 | 0.028 |
| 3279946 | 44306300265 | 44306300275 | 105 m | 0.575 | 0.561 | 0.014 |
| 3279946 | 44306300265 | 44306300425 | 78 m | 0.425 | 0.439 | 0.014 |
| 3279946 | 44306300275 | 44306200025 | 84 m | 0.367 | 0.568 | **0.201** |
| 3279946 | 44306300275 | 44306300365 | 145 m | 0.633 | 0.432 | **0.201** |

## 1.6 Spin-up convergence

Script: `phase1_spinup_convergence.py`.

- Channel cells are those finite in rule 11's burned DEM on the grid.
- The rate is (zs_end − zs_end−6h) / 6 h.
- The mean-discharge level on the channel cells is the skeleton's `zsini.tif`, which carries `mean_level_m` there (TODO B33).

| basin | spin-up | channel cells | max \|dzs/dt\| | median dzs/dt | p95 \|dzs/dt\| | end − mean level: median (p5, p95) |
|---|---|---|---|---|---|---|
| 2433835 | 48 h | 2,091 | 8e-6 m/h | 0.000 m/h | 6e-6 m/h | +0.04 m (+0.00, +0.28) |
| 3279946 | 48 h | 25,220 | 0.166 m/h | +0.001 m/h | 0.010 m/h | +0.28 m (+0.02, +2.07) |

## 1.7 1D–2D gap by angle and width

Script: `phase1_gvf_gap.py`.

- **Gap:** round-1 end-of-run SFINCS level minus the steady 1D level, as the median over each reach's centreline cells.
- **1D level:** `forward_network` on the round-0 bed at the calibration Q, with the calibration sea level at the mouths and rule 10's surface at the other ends.
  - This is the network form of the reach integration in `river_profile.forward_levels`.
  - `forward_levels` covers only each crossing's main stem.
- **Angle:** the reach chord's direction relative to the grid, folded to 0–45°.
- **Width:** SWORD width in cells (dx 100 m).

| basin | reaches | gap median | p5 | p95 | Spearman gap–angle | Spearman gap–width | channel cells | diagonal-only cells |
|---|---|---|---|---|---|---|---|---|
| 2433835 | 8 | +0.76 m | +0.18 m | +1.13 m | +0.19 | −0.55 | 2,112 | 0 |
| 3279946 | 105 | +0.29 m | −0.90 m | +1.41 m | −0.02 | +0.04 | 25,224 | 0 |

- **Figures:** `90_phase1_gvf_gap_scatter.png`.
- **Width classes:** all Chao Phraya reaches are under 3 cells (99) or 3–10 cells (6).
- **Connectivity:** no channel cell connects only diagonally in either basin.

## 1.8 SWOT coverage and datum (SWORD v17c)

Script: `phase1_swot_coverage.py`.

- **Observed node:** `n_obs` ≥ 10 and a finite `wse_obs_p50`.
- **Datum nodes:** observed SWORD nodes on the network (within 500 m) upstream of the upstream-boundary check's influence end (`x_inf_km`, `figs/upstream_boundary_check/`).

| basin | SWORD nodes | observed | reaches with observed nodes | reaches with reliable slope | influence end | datum nodes | obs − static wse | obs − rule 10 (GOCO06s) | obs − rule 10 (EGM2008) | geoid offset |
|---|---|---|---|---|---|---|---|---|---|---|
| 2433835 | 348 | 99.7 % | 7 / 8 | 0 | 56.2 km | 47 | +5.26 m | −1.02 m | −0.95 m | 0.07 m |
| 3279946 | 4,874 | 91.5 % | 97 / 105 | 0 | 276.4 km | 534 | +1.15 m | +0.19 m | +0.27 m | 0.07 m |

- **Datum:** with a geoid offset of only 0.07 m, these numbers cannot tell EGM2008 from GOCO06s.
  - On Chao Phraya the SWOT median level sits 0.2–0.3 m above the rule-10 surface on either datum.
  - On Ebro it sits 1.0 m below it.
  - The static SWORD `wse` disagrees by 1–5 m in both basins.
- **Reliable slopes:** no reach in either basin has a reliable SWOT slope.

## 1.9 Depth sensitivity (Ebro)

Script: `phase1_depth_sensitivity.py run|compare`.

- `river_depth_modelling.depth_scale` set to 0.7 and 1.3, in scratch results roots (`<results_dir>/../scratch_phase1/depth_<scale>/`).
- Rule 11 is force-rerun there, followed by the default build.
- Rule 11 recalibrates the crests on each bed, so the effect combines the bed and its crests.

| basin | depth scale | flooded area | volume | area / baseline | volume / baseline |
|---|---|---|---|---|---|
| 2433835 | 1.0 | 270.22 km² | 0.4141 km³ | 1.000 | 1.000 |
| 2433835 | 0.7 | 266.83 km² | 0.4104 km³ | 0.987 | 0.991 |
| 2433835 | 1.3 | 269.38 km² | 0.4246 km³ | 0.997 | 1.026 |

Chao Phraya was not run (see below).

## Not computed, and why

- **1.0 Chao Phraya: flood extent and volume.** The run was stopped after the spin-up, by decision: the event run is an outcome baseline and not needed for the river depth / GVF / timing diagnostics.
- **1.5 Ebro.** There is no bifurcation on its forcing path.
- **1.9 Chao Phraya.** Its condition (a main run under 2 h) could not be checked, because the main run was not done.
- **1.8 datum.** The geoid offset (0.07 m) is too small in both basins to say which datum SWOT's `wse_obs` uses.
- **1.7.** No reach of either basin is 10 or more cells wide.

## Code changes

| file | function / key | change |
|---|---|---|
| `workflow/src/gvf_solver.py` | `forward_network(..., return_capped=False)` | optional per-node Froude-cap flag (1.4); `tests/unit/test_gvf.py` gains a test |
| `workflow/src/river_depth_calibration.py` | `bifurcation_cross_sections` | cross-section lines across each arm of the seeds' bifurcations (1.5) |
| `config/config.yml`, rule 11 | `river_depth_modelling.crosssections: false` | rule 11 sets those cross-sections in its rounds when true (1.5) |
| `config/config.yml`, rule 11 | `river_depth_modelling.depth_scale: 1.0` | scales the burned depth below the surface, for sensitivity runs (1.9) |
| `analysis/` | `_phase1.py`, `phase1_*.py` | the Phase 1 scripts |

**Default DAG unchanged:** `snakemake -n preprocess|build --forceall --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"` gives 15 / 21 jobs before and after the changes.

## Side effects

- **Logs:** the 1.9 scratch runs write to the repository's shared `logs/` folder, so `logs/2433835/` holds the depth × 1.3 run's logs. The production outputs of both basins are untouched.
- **Run-time figures:** the Chao Phraya run time includes rule 11 writing the new diagnostic files, a one-off cost.
