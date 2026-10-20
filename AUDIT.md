# Audit

Read-only audit of `improve_base_model` at `273ef2d`. `Leuven_et_al/` is deleted in the
working tree and left out. Method: git tree and log, an AST import graph from every entry
point, `snakemake -n --forceall` DAGs for basin 2433835, ruff 0.13.0, and six parallel
read-only reviews (rules, scripts, src, tests, tools/config/root, docs). The key claims
were spot-checked by hand.

## 1. Map

| directory | contents |
|---|---|
| `workflow/` | Snakefile, the entry point, with the targets `preprocess`, `build`, `all` and `adapt` |
| `workflow/rules/` | 26 `.smk` files, 38 rules numbered 00-19; `00_common.smk` holds shared helpers, basins, scenarios and strategies |
| `workflow/scripts/` | 31 Snakemake `script:` files, one per rule step (8.4k lines) |
| `workflow/src/` | 31 library modules imported as `src.*` (19.3k lines) |
| `workflow/archive/` | 3 retired datum-correction files |
| `config/` | config.yml, scenarios.yml, data_catalogue.yml, adaptation_strategies(_manual).yml, measures.yml |
| `tests/unit/` | pytest suite on synthetic data (45 tests) |
| `tests/` | `check_code_health.py`, plus about 20 stand-alone analysis and plot scripts that read pipeline results (`KL_*`, `check_*`, `plot_*`) and a 1.7 MB .tif |
| `tests/archive/` | 3 retired diagnostics |
| `tools/` | 9 manual data-preparation, download and maintenance scripts |
| `figs/dag/` | 4 rule-graph SVGs (tracked although `figs/` is gitignored) |
| `qgis/` | 2 personal QGIS projects |
| `docs/` | not tracked (`/docs/` is gitignored): GVF investigation notes and 8 figures |
| `.github/` | CI (lint and pytest) and the PR template |
| root | README, CONTRIBUTING, CHANGELOG, Reference_memory.txt, environment.yml, pyproject.toml, pre-commit config, plot_dag.py, a hydromt-sfincs bug write-up (.py and .txt twin), a scratchpad probe .yml |

## 2. Reachability

Entry points from CLAUDE.md: the Snakemake targets `preprocess`, `build`, `all` and `adapt`; `tools/*.py`; `plot_dag.py`; `pytest`; `tests/check_code_health.py`.

- **Rules.** All 26 rule files are included by the Snakefile. 28 of the 38 rules are in the `all` DAG for 2433835. The other 10 are the 4 target rules and the 6 adaptation rules (18a, 18b, 18c and 19), which are reachable only through `adapt`. The `adapt` DAG does not resolve on this machine: `advance_1` needs `raw_data/adaptation_ebro/offshore_island.geojson`, which is missing. So adaptation reachability is static only.
- **Scripts.** All 31 are the `script:` of an included rule. None is orphaned.
- **src modules.** 30 of 31 are reached by the pipeline. `src.adaptation_lines` is reached only by `tools/build_adaptation_lines.py` and the tests.
- **Reached by no entry point.**
  - Modules: none.
  - Functions:
    - `src/raster.py:377 clip_ocean_from_topo` (0 callers; only the comments in data_catalogue.yml:140,166 and the docstring at tools/build_deltadtm_mask_vrt.py:14 mention it)
    - `src/raster.py:308 reproject_to_reference_grid`, whose only caller is the dead function above.
  - Files:
    - everything under `workflow/archive/` and `tests/archive/`
    - `config/adaptation_strategies_manual.yml`
    - `scratchpad_probe_active_catalog.yml`
    - `hydromt_sfincs_subgrid_dtype_bugfix.py` and its `.txt` twin
    - `tests/unit/prof`
    - `tests/2433835_landuse_correction_diagnostic.tif`
    - `figs/dag/*.svg`
    - `qgis/*.qgz`
- **Reached only by tests or tools.**
  - `src/plots.py:723 plot_global_protection_map`
  - `src/adaptation_lines.py:394 estuary_and_seal`
  - `src/adaptation_lines.py:429 relabel_estuary`
- **Opt-in branch.** `src/gvf*.py` run only when `river_depth_modelling.method: gvf`; the config default is `modelled` (config.yml:83).

## 3. Classification

Counts: 91 CORE, 40 SUPPORT, 15 STALE, 6 DUPLICATE, 4 UNKNOWN. Line counts and last commits come from git.

| path | lines | last commit | reachable | category | evidence |
|---|---:|---|---|---|---|
| `.gitattributes` | 26 | 2026-04-24 98030b2 | tooling | SUPPORT | git config; no `*.qgz binary` rule |
| `.github/pull_request_template.md` | 31 | 2026-07-03 da059d7 | CI | SUPPORT | PR checklist; references check_code_health.py |
| `.github/workflows/ci.yml` | 62 | 2026-10-20 c46e13e | CI | SUPPORT | lint + pytest job; pip set (:61) lacks scipy, needed by tests/unit/test_gvf.py via gvf_solver.py:56 |
| `.gitignore` | 32 | 2026-08-03 2f807ac | tooling | SUPPORT | typo `scatchpat_*` (:7) lets scratchpad_probe through; ignores /docs/ and figs/ which hold content |
| `.pre-commit-config.yaml` | 61 | 2026-07-14 c175b76 | hook | SUPPORT | global exclude skips workflow/scripts/ (:7) and 4 non-existent dirs |
| `CHANGELOG.md` | 2144 | 2026-10-20 273ef2d | doc | SUPPORT | 2144 lines of per-change rationale (history) |
| `CLAUDE.md` | 39 | 2026-10-20 c46e13e | doc | CORE | project facts; 'rules 01-12' copied from Snakefile:200, no rule 12 exists |
| `CONTRIBUTING.md` | 360 | 2026-10-09 935ce9f | doc | SUPPORT | only real install guide; stage range 00-16 outdated (:11) |
| `README.md` | 13 | 2026-05-01 a67b742 | doc | SUPPORT | 13 lines: setup stub, no description or run commands |
| `Reference_memory.txt` | 3409 | 2026-10-20 c46e13e | doc | SUPPORT | pipeline listed to rule 17, no adaptation, machine conda path; holds unique domain facts (section 5) |
| `config/adaptation_strategies.yml` | 74 | 2026-10-19 b60ed97 | yes | CORE | config.yml:4 `adaptation.strategies_file` |
| `config/adaptation_strategies_manual.yml` | 355 | 2026-10-15 be4fdcd | no | STALE | 0 code refs; per-scale hand-written list superseded by scale-1 families in adaptation_strategies.yml + src/adaptation_scaling.py |
| `config/config.yml` | 299 | 2026-10-20 273ef2d | yes | CORE | Snakefile configfile; `river_depth_modelling.gvf.target_discharge` (:85) never read |
| `config/data_catalogue.yml` | 928 | 2026-10-13 e68f469 | yes | CORE | 22 entries; `river_basins` (:29) unreferenced |
| `config/measures.yml` | 167 | 2026-10-15 be4fdcd | yes | CORE | config.yml:5 `adaptation.measures_file` |
| `config/scenarios.yml` | 105 | 2026-10-19 b60ed97 | yes | CORE | `scenarios_file`; scenario_params in 00_common.smk |
| `environment.yml` | 544 | 2026-10-09 18e98f0 | yes | CORE | pinned win-64 export; no ruff/pyright/pydocstyle |
| `figs/dag/build_dag.svg` | 260 | 2026-06-09 d0ebfe2 | no | STALE | plot_dag.py output of 2026-06-09, predates rules 07a/18-19; under ignored figs/ |
| `figs/dag/build_rulegraph.svg` | 259 | 2026-06-09 d0ebfe2 | no | STALE | plot_dag.py output of 2026-06-09, predates rules 07a/18-19; under ignored figs/ |
| `figs/dag/preprocess_dag.svg` | 260 | 2026-06-09 d0ebfe2 | no | STALE | plot_dag.py output of 2026-06-09, predates rules 07a/18-19; under ignored figs/ |
| `figs/dag/preprocess_rulegraph.svg` | 259 | 2026-06-09 d0ebfe2 | no | STALE | plot_dag.py output of 2026-06-09, predates rules 07a/18-19; under ignored figs/ |
| `hydromt_sfincs_subgrid_dtype_bugfix.py` | 110 | 2026-07-14 c175b76 | no | UNKNOWN | bug write-up/repro, 0 imports; fix lives in a personal clone, not in pinned hydromt-sfincs 2.0.0rc2 (environment.yml:520) |
| `memory_subgrid_dtype_bug_and_fix.txt` | 110 | 2026-07-26 910c9ab | no | DUPLICATE | twin of hydromt_sfincs_subgrid_dtype_bugfix.py (line endings differ only) |
| `plot_dag.py` | 108 | 2026-06-09 d0ebfe2 | manual | SUPPORT | writes figs/dag/*.svg; CLAUDE.md entry point |
| `pyproject.toml` | 31 | 2026-10-20 c46e13e | tooling | SUPPORT | ruff/pytest/pyright config; pyright exclude names 5 non-existent files |
| `qgis/domain_choice.qgz` | 118 | 2026-06-09 d0ebfe2 | no | UNKNOWN | personal QGIS project, 0 refs |
| `qgis/inspect weir_depth protection.qgz` | 139 | 2026-07-26 910c9ab | no | UNKNOWN | personal QGIS project, 0 refs, space in name |
| `scratchpad_probe_active_catalog.yml` | 6 | 2026-08-03 2f807ac | no | STALE | 6-line probe with D:\ paths, 0 refs; meant to be ignored (gitignore typo) |
| `tests/2433835_landuse_correction_diagnostic.tif` | 18 | 2026-08-07 0231172 | no | STALE | 1.67 MB data file, 0 refs |
| `tests/KL_analysis_waterlevel_animation.py` | 195 | 2026-10-09 935ce9f | stand-alone | SUPPORT | reads results; basin hard-coded |
| `tests/KL_check_adaptation_scaling.py` | 88 | 2026-10-15 be4fdcd | stand-alone | SUPPORT | checks src.adaptation_scaling; D:/ path at :25 |
| `tests/KL_csi.py` | 318 | 2026-10-09 935ce9f | stand-alone | SUPPORT | PRE/POST CSI from results; no pipeline equivalent |
| `tests/KL_domain_figures.py` | 269 | 2026-10-09 935ce9f | stand-alone | SUPPORT | paper figures; docstring basin (:24) differs from code (:49) |
| `tests/KL_flood_map_pre_post.py` | 169 | 2026-10-09 935ce9f | stand-alone | DUPLICATE | near-twin of tests/KL_flood_maps.py (depth panel) |
| `tests/KL_flood_maps.py` | 228 | 2026-10-09 935ce9f | stand-alone | SUPPORT | attribution panel overlaps src.attribution_plot.generate_attribution_maps |
| `tests/KL_gtsm_storm_tide.py` | 714 | 2026-10-09 935ce9f | stand-alone | STALE | ported to src/storm_tide_hydrograph.py (its :8 says so) + scripts 07b/07c |
| `tests/KL_results_figures.py` | 694 | 2026-10-13 80f0393 | stand-alone | SUPPORT | paper figures from results, 2 basins hard-coded |
| `tests/archive/_diag_lagoon_timeseries.py` | 47 | 2026-10-09 935ce9f | no | STALE | archived one-off diagnostic, D:/ experiment paths (:11-12) |
| `tests/archive/check_diluvium.py` | 245 | 2026-10-09 935ce9f | no | STALE | archived; excluded from lint |
| `tests/archive/sync_deltadtm.py` | 237 | 2026-10-09 935ce9f | no | STALE | archived; excluded from lint |
| `tests/check_code_health.py` | 90 | 2026-07-03 5efcf4e | documented check | CORE | ci.yml, PR template, CLAUDE.md |
| `tests/check_crest_calibration.py` | 286 | 2026-10-09 935ce9f | stand-alone | SUPPORT | re-derives rule 10 crest bookkeeping from results |
| `tests/check_surge_shape_duration.py` | 643 | 2026-10-09 935ce9f | stand-alone | UNKNOWN | diagnostic for KL_gtsm_storm_tide surge shapes; stale if that is |
| `tests/inspect_grdc_correlation.py` | 192 | 2026-10-09 935ce9f | stand-alone | SUPPORT | GRDC vs river_forcing.nc; ref only in Reference_memory |
| `tests/plot_calibration_round_profiles.py` | 150 | 2026-10-09 935ce9f | stand-alone | DUPLICATE | CLI around src.plots.plot_calibration_round_profiles, already called by scripts/10 |
| `tests/plot_coastal_protection_weir.py` | 125 | 2026-10-09 935ce9f | stand-alone | DUPLICATE | re-implements src.plots.plot_coastal_protection_weir (rendered by rule 13) |
| `tests/plot_global_protection_levels.py` | 71 | 2026-07-14 c175b76 | stand-alone | SUPPORT | only caller of src.plots.plot_global_protection_map |
| `tests/plot_overtopping_volume.py` | 215 | 2026-10-09 935ce9f | stand-alone | SUPPORT | reads results |
| `tests/plot_surge_vs_dike_crest.py` | 293 | 2026-10-09 935ce9f | stand-alone | SUPPORT | reads results; uses src.surge |
| `tests/unit/conftest.py` | 41 | 2026-10-20 c46e13e | pytest | CORE | fixtures, sys.path |
| `tests/unit/prof` | 307 | 2026-10-20 c46e13e | no | STALE | 66 KB cProfile dump with a C:\Users path, 0 refs |
| `tests/unit/test_accumulate_discharge.py` | 117 | 2026-10-20 c46e13e | pytest | CORE | 10 tests, src.river_network |
| `tests/unit/test_gvf.py` | 435 | 2026-10-20 c46e13e | pytest | CORE | 20 tests, gvf_solver/gvf_target |
| `tests/unit/test_network_qa.py` | 125 | 2026-10-20 c46e13e | pytest | CORE | 10 tests, src.river_network |
| `tests/unit/test_round0_steady.py` | 54 | 2026-10-20 c46e13e | pytest | CORE | 5 tests, river_depth_calibration, sfincs_run |
| `tools/build_adaptation_lines.py` | 327 | 2026-10-13 e68f469 | manual | SUPPORT | only consumer of src.adaptation_lines; hidden per-basin step after rule 13 (own docstring) |
| `tools/build_deltadtm_mask_vrt.py` | 203 | 2026-10-09 935ce9f | manual | DUPLICATE | 4 functions twin tools/build_esa_worldcover_vrt.py; docstring (:14) cites dead clip_ocean_from_topo |
| `tools/build_esa_worldcover_vrt.py` | 282 | 2026-10-09 935ce9f | manual | SUPPORT | one-off data prep; referenced by rule 02b, catalogue |
| `tools/download_cds_waterlevel.py` | 76 | 2026-10-09 935ce9f | manual | SUPPORT | one-off download; no main(), runs on import |
| `tools/download_fathomdem.py` | 304 | 2026-07-14 c175b76 | manual | SUPPORT | one-off download; catalogue:60 |
| `tools/download_glofas.py` | 53 | 2026-10-09 935ce9f | manual | SUPPORT | one-off download; no main(), runs on import |
| `tools/prepare_modified_sword_dataset.py` | 170 | 2026-10-09 935ce9f | manual | SUPPORT | builds the production river_network gpkg |
| `tools/replot_event_figures.py` | 144 | 2026-10-09 935ce9f | manual | DUPLICATE | repeats plotting block of scripts/16_run_event.py:48-100 |
| `tools/sync_environment.py` | 80 | 2026-07-03 da059d7 | hook | SUPPORT | pre-commit sync-environment-yml |
| `workflow/Snakefile` | 231 | 2026-10-20 c46e13e | entry | CORE | targets preprocess/build/all/adapt |
| `workflow/archive/datum_correction/03a_datum_correction_excerpt.py` | 137 | 2026-06-09 d0ebfe2 | no | STALE | 0 refs outside archive; superseded by scripts/05a_get_elevation.py:22-31 |
| `workflow/archive/datum_correction/compute_geoid_offset_arr.py` | 88 | 2026-06-09 d0ebfe2 | no | STALE | 0 refs outside archive; superseded by scripts/05a_get_elevation.py:22-31 |
| `workflow/archive/datum_correction/plot_datum_correction_delta.py` | 93 | 2026-06-09 d0ebfe2 | no | STALE | 0 refs outside archive; superseded by scripts/05a_get_elevation.py:22-31 |
| `workflow/rules/00_common.smk` | 456 | 2026-10-15 be4fdcd | yes | CORE | Snakefile:10; BASINS/SCENARIOS/STRATEGIES, path helpers |
| `workflow/rules/01_separate_deltas.smk` | 9 | 2026-08-04 2e14535 | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/02_modeldomain.smk` | 15 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/02b_prepare_landuse.smk` | 65 | 2026-09-14 556b8a4 | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/03_land_polygons.smk` | 29 | 2026-09-14 556b8a4 | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/04_protection_levels.smk` | 35 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/05_static_terrain_data.smk` | 135 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/06_river_network.smk` | 47 | 2026-09-14 556b8a4 | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/07_boundary_forcings.smk` | 104 | 2026-10-20 c46e13e | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/07a_storm_tide_hydrographs.smk` | 71 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/08_clean_river_network.smk` | 21 | 2026-10-20 c46e13e | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/08c_build_sfincs_grid.smk` | 28 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/09_condition_elevation.smk` | 50 | 2026-10-20 c46e13e | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/09b_grid_align_landuse.smk` | 50 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/10_depth_estimation_modelled.smk` | 175 | 2026-10-20 c46e13e | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/11_testing.smk` | 84 | 2026-10-20 c46e13e | yes | SUPPORT | Snakefile:41; 2 validation-plot rules in `all`, outputs consumed by nothing |
| `workflow/rules/13_build_sfincs.smk` | 97 | 2026-10-19 b60ed97 | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/13_build_sfincs_skeleton.smk` | 126 | 2026-10-20 c46e13e | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/14_run_spinup.smk` | 59 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/15_sanity_checks.smk` | 34 | 2026-10-09 935ce9f | yes | SUPPORT | Snakefile:45; diagnostics in all-DAG |
| `workflow/rules/16_run_event.smk` | 49 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/17_flood_metrics.smk` | 27 | 2026-10-09 935ce9f | yes | CORE | included by Snakefile; in all-DAG for 2433835 |
| `workflow/rules/18a_adapt_pre.smk` | 177 | 2026-10-19 b60ed97 | adapt (static) | CORE | included by Snakefile; reached via `adapt` |
| `workflow/rules/18b_adapt_post.smk` | 44 | 2026-10-15 be4fdcd | adapt (static) | CORE | included by Snakefile; reached via `adapt` |
| `workflow/rules/18c_attribution_mask.smk` | 44 | 2026-10-09 935ce9f | adapt (static) | CORE | included by Snakefile; reached via `adapt` |
| `workflow/rules/19_adapt_compare.smk` | 25 | 2026-08-18 36422a3 | adapt (static) | CORE | included by Snakefile; reached via `adapt` |
| `workflow/scripts/01_separate_deltas.py` | 27 | 2026-06-09 d0ebfe2 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/02_determine_model_domain.py` | 70 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/02b_prepare_landuse.py` | 65 | 2026-09-14 556b8a4 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/03_get_land_polygons.py` | 49 | 2026-09-14 556b8a4 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/04_get_protection_levels.py` | 72 | 2026-08-07 0231172 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/05a_get_elevation.py` | 601 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/05b_get_landuse.py` | 105 | 2026-09-14 556b8a4 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/05c_get_roughness.py` | 66 | 2026-09-14 556b8a4 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/06_get_river_network.py` | 57 | 2026-09-14 556b8a4 | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/06b_plot_river_network.py` | 27 | 2026-09-14 556b8a4 | yes | SUPPORT | rule plot_river_network; figure only |
| `workflow/scripts/07_get_boundary_forcings.py` | 819 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/07a_select_surge_stations.py` | 54 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/07b_extract_gtsm_series.py` | 68 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/07c_storm_tide_hydrographs.py` | 167 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/08_clean_river_network.py` | 225 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/08c_build_sfincs_grid.py` | 47 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/09_condition_elevation.py` | 119 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/09b_grid_align_landuse.py` | 157 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/10_depth_estimation_modelled.py` | 2105 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/11_test_upstream_boundary.py` | 528 | 2026-09-14 556b8a4 | yes | SUPPORT | rule test_upstream_boundary; PNG only |
| `workflow/scripts/11b_bifurcation_calibration_options.py` | 399 | 2026-10-20 c46e13e | yes | SUPPORT | plots only; clip_and_preclean (:108) is a drifted copy of 06+08 cleaning |
| `workflow/scripts/13_build_sfincs.py` | 582 | 2026-10-20 273ef2d | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/13_build_sfincs_skeleton.py` | 723 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/14_run_spinup.py` | 349 | 2026-10-20 c46e13e | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/15_sanity_checks.py` | 132 | 2026-10-09 935ce9f | yes | SUPPORT | rule sanity_checks |
| `workflow/scripts/16_run_event.py` | 199 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/17_flood_metrics.py` | 60 | 2026-10-09 935ce9f | yes | CORE | `script:` of a rule; in all-DAG for 2433835 |
| `workflow/scripts/18a_adapt_pre.py` | 256 | 2026-10-15 be4fdcd | adapt (static) | CORE | `script:` of a rule; reached via `adapt` |
| `workflow/scripts/18b_adapt_post.py` | 147 | 2026-10-15 be4fdcd | adapt (static) | CORE | `script:` of a rule; reached via `adapt` |
| `workflow/scripts/18c_attribution_mask.py` | 79 | 2026-10-09 935ce9f | adapt (static) | CORE | `script:` of a rule; reached via `adapt` |
| `workflow/scripts/19_adapt_compare.py` | 66 | 2026-08-19 48a0bd1 | adapt (static) | CORE | `script:` of a rule; reached via `adapt` |
| `workflow/src/adaptation_lines.py` | 439 | 2026-10-13 e68f469 | tools only | SUPPORT | imported only by tools/build_adaptation_lines.py |
| `workflow/src/adaptation_method_post.py` | 1051 | 2026-10-15 be4fdcd | adapt (static) | CORE | 18b; shares 10 function names with _pre, validate_measure_params AST-identical |
| `workflow/src/adaptation_method_pre.py` | 1759 | 2026-10-15 be4fdcd | adapt (static) | CORE | 18a dispatch_rules |
| `workflow/src/adaptation_scaling.py` | 206 | 2026-10-15 be4fdcd | adapt (static) | CORE | strategy scale expansion (00_common.smk) |
| `workflow/src/attribution_plot.py` | 153 | 2026-09-18 2d9ff31 | adapt (static) | SUPPORT | plotting for 18c |
| `workflow/src/domain.py` | 78 | 2026-07-14 c175b76 | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/extreme_values.py` | 1488 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/geometry.py` | 133 | 2026-07-26 910c9ab | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/grid.py` | 179 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/gtsm.py` | 146 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/gvf.py` | 232 | 2026-10-20 c46e13e | yes | CORE | opt-in `river_depth_modelling.method: gvf` (config default `modelled`, config.yml:83) |
| `workflow/src/gvf_solver.py` | 636 | 2026-10-20 c46e13e | yes | CORE | via gvf; unit-tested |
| `workflow/src/gvf_target.py` | 821 | 2026-10-20 c46e13e | yes | CORE | via gvf; unit-tested |
| `workflow/src/io.py` | 65 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/landuse.py` | 855 | 2026-10-20 c46e13e | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/log.py` | 33 | 2026-06-09 d0ebfe2 | yes | SUPPORT | logging setup |
| `workflow/src/plots.py` | 3145 | 2026-10-20 273ef2d | yes | SUPPORT | 3145 lines, 18 functions >100 lines |
| `workflow/src/postprocessing.py` | 531 | 2026-10-20 c46e13e | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/profiling.py` | 123 | 2026-06-09 d0ebfe2 | yes | SUPPORT | reads the snakemake global inside library code (:36,49) |
| `workflow/src/protection_levels.py` | 203 | 2026-09-14 556b8a4 | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/protection_weir.py` | 991 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/raster.py` | 574 | 2026-10-09 935ce9f | yes | CORE | clip_ocean_from_topo (:377) and reproject_to_reference_grid (:308) dead |
| `workflow/src/river_burn.py` | 836 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/river_conditioning.py` | 303 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/river_depth_calibration.py` | 290 | 2026-10-20 c46e13e | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/river_forcing.py` | 1144 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/river_network.py` | 1100 | 2026-10-20 c46e13e | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/sfincs_restart.py` | 161 | 2026-09-11 69343c1 | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/sfincs_run.py` | 223 | 2026-10-20 c46e13e | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/storm_tide_hydrograph.py` | 256 | 2026-10-09 935ce9f | yes | CORE | imported transitively by pipeline scripts |
| `workflow/src/surge.py` | 1185 | 2026-10-19 b60ed97 | yes | CORE | imported transitively by pipeline scripts |

## 4. Quality findings (ranked by impact; no fixes)

**Correctness risks**
1. `workflow/src/extreme_values.py:554,601,935` | `except Exception: pass` makes bootstrap GEV/GPD refits and the KS test fail silently. More broad excepts that fall back silently: extreme_values.py:840,937,998,1016, plots.py:2926 and postprocessing.py:383 (pixel area becomes NaN).
2. Reach-ID normalisation is written 4 times with different bodies: `scripts/08_clean_river_network.py:25 _norm_id`, `scripts/13_build_sfincs_skeleton.py:569 _norm_rid`, `src/river_forcing.py:42 _norm_reach_id` and `src/river_network.py:63 normalize_reach_id`. They give different results for "nan" and "<NA>".
3. `workflow/scripts/11b_bifurcation_calibration_options.py:108 clip_and_preclean` says it "mirrors 06 + 08" but leaves out `canal_excluded_arms` and `outward_seed_reaches` (08:14,20). It has already drifted.
4. `workflow/Snakefile:134-154` | the `run_event` and `flood_metrics` targets are gated on `spinup.enabled`. With spin-up off, `build` silently runs no event.
5. `.github/workflows/ci.yml:61` | CI pip-installs neither scipy (imported at `src/gvf_solver.py:56`, which `tests/unit/test_gvf.py:8` loads) nor pyextremes/matplotlib (`src.gvf` → river_forcing → extreme_values). The unit-test job very likely fails on import. This is inferred, not run.
6. `.pre-commit-config.yaml:7` | the global exclude skips `workflow/scripts/`, so none of the 31 rule scripts are linted or docstring-checked when you commit.

**Duplicated logic**
7. The isolated-sea-cell zsini dry-out is copied line for line: `scripts/10_depth_estimation_modelled.py:644-678` and `scripts/13_build_sfincs_skeleton.py:436-492`.
8. sfincs.inp is written by hand twice, with key sets that have drifted apart: `scripts/13_build_sfincs.py:414-476` and `scripts/14_run_spinup.py:226-277`.
9. The discharge-point placement sequence appears 3 times: `13_build_sfincs.py:377-393`, `14_run_spinup.py:~201` and `10_depth_estimation_modelled.py:899-912`.
10. Rule params are copied:
    - `rules/18a_adapt_pre.smk:80-100` repeats 15 config keys of `13_build_sfincs.smk:33-90` (18a:101 admits only two differ).
    - `18a:140-147` repeats `16_run_event.smk:30-37`.
11. `src/adaptation_method_pre.py` and `adaptation_method_post.py` share 10 function names. `validate_measure_params` is AST-identical in both, and `_is_current_protection` differs by 11 lines.
12. Between `src/gvf.py` and `src/river_depth_calibration.py`:
    - `gvf.py:36 mean_seed_discharge` duplicates `river_depth_calibration.py:41 build_calibration_seed_discharge`.
    - `gvf.py:202 depth_segments` duplicates `river_depth_calibration.py:257 cell_depth_segments`.
13. A Kahn topological sort is written 4 times: `river_conditioning.py:133`, `river_network.py:524`, `gvf_target.py:150` and `gvf_target.py:547`.
14. MDT loading is re-implemented in `scripts/05a_get_elevation.py:359-466`; `src/surge.py:581 load_mdt` and `:603 apply_mdt_correction` already exist.
15. `scripts/13_build_sfincs.py:478-582` draws the forcing plot by hand, while `src/plots.py:982 plot_forcing_timeseries` exists.
16. Raster and polyline I/O is spread out:
    - Readers: `plots.py:226`, `landuse.py:200`, `raster.py:423`, plus inline `rasterio.open` in 6 modules.
    - Weir/polyline readers and writers: `adaptation_scaling.py:54,73`, `adaptation_method_pre.py:255` and `sfincs_run.py:104`.
17. Twins in tools and tests:
    - `tools/build_deltadtm_mask_vrt.py` and `build_esa_worldcover_vrt.py` (4 functions)
    - `tools/replot_event_figures.py` and `scripts/16_run_event.py:48-100`
    - `tests/plot_coastal_protection_weir.py` and `src/plots.py`
    - `tests/KL_flood_map_pre_post.py` and `KL_flood_maps.py`
18. Path stems are repeated by hand:
    - `preprocessing_inputs/domain/{basin_id}_*` about 140 times in rules and 17 in the Snakefile
    - `{basin_id}/runs/{scenario}/adaptation/pre/{strategy}` 23 times in 18a
    - about 60 `expand(..., basin_id=BASINS)` lines in Snakefile:56-154

**Structure and size**
19. `scripts/10_depth_estimation_modelled.py` | 2105 lines: 907 of module-level code, 17 helpers that close over globals, and a 240-line module docstring. The round runner, the I/O readers (955-1143) and the forcing builder (1828-1905) are library logic.
20. Functions over about 100 lines (46 in total):
    - `river_burn.py:70 burn_river_channel` (362)
    - `adaptation_method_pre.py:702 apply_water_retention` (354)
    - `adaptation_method_pre.py:376 apply_NbS_land_reclamation` (324)
    - `protection_weir.py:681 build_coastal_protection_weir` (311)
    - `extreme_values.py:696 analyse_cell` (253)
    - `river_conditioning.py:68` (236)
    - `river_forcing.py:518` (231)
    - `surge.py:984` (202)
    - `plots.py` has 18 of them in 3145 lines.
21. Scripts with no `main()` and hundreds of module-level lines: 07 (607), 13_skeleton (423), 05a (381), 13_build_sfincs (346), 11 (340), 14 (209). Late imports (E402): 05a:280,369,410,570; 13_build_sfincs:486-489; 13_skeleton:558; 14:291,293.
22. Library code doing I/O or reading globals:
    - `profiling.py:36,49` reads `snakemake.config`.
    - `adaptation_method_post.py` has 8 raster writes inside its `apply_*` functions, landuse.py 4, river_conditioning.py 1.
    - `extreme_values.py:1187,1331` draws plots.
23. print instead of logging in library code: adaptation_method_post (18), adaptation_method_pre (9), attribution_plot (4), grid (3), surge (2), sfincs_run (1), profiling (1).
24. Private src functions imported across modules: `08:11` and `11:72` use `src.river_network._as_linestring`; `05a:92` uses `_tile_intersects`.

**Hard-coded paths and constants**
25. Absolute paths:
    - `tests/KL_check_adaptation_scaling.py:25` (`D:/GCFM_UU/raw_data`)
    - `tests/archive/_diag_lagoon_timeseries.py:11-12`
    - `scratchpad_probe_active_catalog.yml:2,5` (tracked because of the `.gitignore:7` typo `scatchpat_*`)
    - `hydromt_sfincs_subgrid_dtype_bugfix.py:6`
    - `tests/unit/prof` (binary)
    - `Reference_memory.txt:9-11`
26. Magic constants:
    - `-9999.0` nodata written 28 times across 11 scripts, with `NODATA` redefined at 05a:553, 05b:71 and 10:1750
    - `111_320.0` m per degree (05a:163)
    - the 1e6 sentinel (10:650, 13_skeleton:452)
    - a 0.25 h surge step (10:1858)
    - `_SURGE_RPS` and the river RP bounds 2/1000 (00_common.smk:149-150)
    - a strategy-scale fallback that hides config (00_common.smk:290)
    - the scenario `"default"` hard-coded in a basin-level rule (07_boundary_forcings.smk:62)
27. Hard-coded basin IDs in the `tests/KL_*` scripts (e.g. KL_flood_maps.py:36, KL_results_figures.py:28,357, KL_csi.py:36).

**Dead branches and dead config**
28. Dead config:
    - `config.yml:85 river_depth_modelling.gvf.target_discharge` is never read.
    - `data_catalogue.yml:29 river_basins` is never referenced; its parameter `river_basins_path` is left behind at `plots.py:285`.
    - `measures.yml` `type` and `description` are never read.
29. Branches the current config never takes:
    - hydrograph off (Snakefile:73-75)
    - spin-up off (Snakefile:134-154; 13_build_sfincs.smk:24-27)
30. Dead code in comments and TODOs:
    - `18a_adapt_pre.py:76,79` keeps commented-out measures.
    - `18b_adapt_post.py:87 catalog_path=None` is unused by its own TODO.
    - Open TODOs at `07:583`, `07c:24` and `18a:179`.
31. Exclude lists name files that don't exist: pyproject.toml pyright exclude (5 files) and `.pre-commit-config.yaml:1-9` (4 directories).

**Naming: one concept, several names**
32. Return period: `rp_yr` (83 in src), `return_period` (37), `rp` (33), `surge_rp`, `surge_return_period` (07.smk:62), `design_rp_surge_yr` (13.smk:55), `coastal_rp_yr`/`riverine_rp_yr`, `rp_fl`, `rp_bf`. The `rp_years` convention is used nowhere.
33. River depth: `rivdph`, `river_depth*`, `depth_m`, `h` (gvf_solver).
34. Grid spacing: `resolution`, `resolution_m`, `work_res_m`, `dx`.
35. Reach ID: `rid`, `reach_id`.
36. Model area: `basin`, `delta`, `domain`.
37. Sea mask: `_zsini_sea_cells_on_grid.tif`, `sea_mask.tif`, `_sea_mask.tif`, `sea_mask_corrected.tif`.
38. Rule and file names that mislead:
    - `10_depth_estimation_modelled` also runs the `gvf` method.
    - `11_testing.smk` holds production targets.
    - `09_condition_elevation.py` runs rule `enforce_river_monotonicity`.
    - Two rule files share the number 13.
39. One threshold under two names: `15_sanity_checks.py:44` calls the param `min_inundation_depth_m` `threshold_m`.

**Documentation in code (section 5)**
40. History wording in code comments and docstrings:
    - about 31 history comments in rules (e.g. 18a.smk:2,9)
    - about 20 in scripts (e.g. 13_build_sfincs.py:478-485 "fixed 2026-08-05")
    - plan and date references in the docstrings of gvf.py:1-12, gvf_target.py and gvf_solver.py
41. Docstrings that contradict the code:
    - `10_depth_estimation_modelled.py:1-40` ("only way" depth is set; no mention of gvf)
    - `18c_attribution_mask.py:1-4` ("thin wrapper", yet it computes volumes)
    - `15_sanity_checks.py:1-25` (animation not mentioned)
    - `Snakefile:159-164` (`plot_inundation_ratio` is not in `_ADAPT_OUTPUTS`)
    - `Snakefile:200` and CLAUDE.md ("rules 01-12"; there is no rule 12)
    - `workflow/archive/.../03a_datum_correction_excerpt.py:7-10`
    - `tests/KL_check_adaptation_scaling.py:11`, `tests/KL_domain_figures.py:24`

## 5. Docs state

**Outdated compared with the code**
- `README.md` | 13 lines: 6 blank lines, then a hydromt_sfincs clone step. No description, commands, targets, outputs or data.
- `Reference_memory.txt` (3409 lines) | Coverage and history:
  - lists the pipeline only to rule 17 (:122-136), with no 02b, 07a, 13_skeleton or 18-19
  - has no `adapt` target and 0 mentions of adaptation
  - says there are no pytest suites (:2326)
  - full of "REMOVED/fixed <date>" history (e.g. :2959)
- `Reference_memory.txt` | Wrong paths and versions:
  - FathomDEM script path `tests/download_fathomdem.py` (:522), which is really `tools/download_fathomdem.py`; the same stale path is in data_catalogue.yml:60
  - names `8_polygons.json` (:561, 1204), but the catalogue uses `9_polygons.gpkg`
  - mentions quadtree (:3036), which was removed
  - asks for a separate pyshtools/boule install (:97-99), but both are in environment.yml
  - names HydroMT-SFINCS `2.0.0.dev0` (:3051), but the pinned version is `2.0.0rc2`
  - gives the run command with a personal miniforge path (:9-11)
- `CONTRIBUTING.md` | Stale details:
  - stages "00-16" (:11) and "rule 17" as future work (:17)
  - names the current branches (:42-44)
  - the PR checklist (:144-149) leaves out pytest and check_code_health
- `Snakefile:200` and `CLAUDE.md` | "rules 01-12".
- The hydromt_sfincs install contradicts itself:
  - README says to clone it editable.
  - environment.yml:520 pins pip `2.0.0rc2`.
  - The subgrid-dtype memory fix exists only in a personal clone.
- `CHANGELOG.md` (2144 lines) is current, but it is history that section 5 says belongs in git.

**What a new contributor could not run**
1. The README has no quick start. The commands exist only in CLAUDE.md, Reference_memory and the Snakefile docstrings.
2. Raw data: nothing says what to download, from where, which version or licence (22 catalogue entries), or the directory layout under `$GCFM_RAW_DATA_ROOT`. FathomDEM needs a restricted Zenodo share link.
3. SFINCS: no version, source or licence (CONTRIBUTING:233 says only "separately licensed").
4. environment.yml is a win-64 export (no Linux or HPC solve). The env vars are documented only in PowerShell.
5. On large basins, a fresh install lacks the hydromt-sfincs subgrid memory fix.
6. `adapt` needs per-basin raw adaptation geometries (e.g. `adaptation_ebro/offshore_island.geojson`) and a manual `tools/build_adaptation_lines.py` step. Neither is documented.
7. `/docs/` is gitignored, so the docs pages would never be shared.
8. There is no LICENSE or CITATION.cff, and no description of the outputs under `$GCFM_RESULTS_DIR/{basin_id}/`.

**Domain facts to keep in any doc trim** (from Reference_memory; found nowhere else as prose)
- **Grid orientation:** SfincsModel grids are y-ascending, while `sfincs_grid.json` is stored y-descending. Reading it directly flips the rows.
- **Grid size:** the default dx is 100 m with per-basin overrides (4267691: 150 m); subgrid = dx/10.
- **Calibration:** rule 10 builds and runs a disposable SFINCS model 3 times to set channel depth and per-edge weir crests.
- **MDT (CNES-CLS22):**
  - used twice: GEBCO→GOCO06s in 05a, and the surge correction in 07
  - NaN over land, so it must be extrapolated
- **FLOPROS:**
  - fallback order: own hazard → other hazard → `default_rp_yr`
  - about 74% of units have a riverine value and about 25% a coastal one
- **SWORD:** the modified file is the production default; v17c is used only by 11b. Bifurcation splits by width only.
- **GRDC bias correction:** covers the test basins only (83 stations). Other basins get none.
- **Failed jobs:** Snakemake cleanup deletes the completed SFINCS outputs of a failed job.
- **hydromt-sfincs and SWORD quirks:**
  - open the model with `w+`, never `r+`
  - normalise `reach_id` with `str(int(float(x)))`
- **AR6 SLR:** Zenodo 10.5281/zenodo.5914710, in mm. `storevel` is hard-wired to 0.

## 6. Test state

There are 45 unit tests on synthetic data. They test 5 src modules directly (river_network, gvf_solver, gvf_target, river_depth_calibration, and two functions of sfincs_run). No pipeline script is exercised.

**CORE modules with no assertions** (imported only as a side effect, or not at all):
- `src`: domain, grid, io, raster, geometry, extreme_values, gtsm, surge, storm_tide_hydrograph, landuse, protection_levels, protection_weir, river_burn, river_conditioning, river_forcing, gvf, postprocessing, sfincs_restart, adaptation_method_pre, adaptation_method_post, adaptation_scaling.
- `sfincs_run` beyond `parse_sfincs_inp` and `set_sfincs_inp_keys`.
- `workflow/scripts`: all 31. The library logic inside scripts 07, 10, 13 and 14 (findings 7-9, 19) is untestable as written.
- No quick-look plotting helper is shared across stages. Each script draws its own figures, often inline.

**Unit-test hygiene**
- Exact-float `==` asserts on pass-through values: test_network_qa.py:38,44,45,51; test_accumulate_discharge.py:74-75; test_round0_steady.py:18; test_gvf.py:96.
- Two test names don't state the check: `test_network_qa.py:115`, `test_accumulate_discharge.py:22`.
- No marker exists for data-dependent tests. That is fine today, but the stand-alone scripts in `tests/` would run their module-level code if they were ever collected.

## 7. Questions for you

1. **Is `workflow/archive/datum_correction/` safe to delete?** Guess: yes. 05a does the datum correction today.
2. **Is `tests/archive/` safe to delete?** Guess: yes. These are one-off diagnostics, already excluded from lint.
3. **Is `config/adaptation_strategies_manual.yml` a reference copy or dead?** Guess: dead, superseded by the scale-1 families (2026-10-16). It's KL's file, so confirm with KL.
4. **Should the `tests/KL_*` and `tests/plot_*`/`check_*` analysis scripts stay in `tests/`?** Guess: move them to `analysis/` (or delete the DUPLICATE/STALE ones), so `tests/` holds only tests.
5. **Can `tests/KL_gtsm_storm_tide.py` and `tests/check_surge_shape_duration.py` go?** Guess: yes, ported into `src/storm_tide_hydrograph.py`.
6. **What should happen to the hydromt-sfincs subgrid fix?** It is in a personal clone only (`hydromt_sfincs_subgrid_dtype_bugfix.py` + .txt). Guess: open an upstream PR or pin a fork commit in environment.yml, then delete both root files.
7. **Delete `scratchpad_probe_active_catalog.yml`, `tests/unit/prof` and `tests/2433835_landuse_correction_diagnostic.tif`, and fix the `.gitignore` typo?** Guess: yes to all.
8. **What about `figs/dag/*.svg` (stale, under an ignored path)?** Guess: untrack them and regenerate on demand with `plot_dag.py`.
9. **Should `qgis/*.qgz` be tracked?** Guess: no. They are personal projects; untrack them.
10. **Should `/docs/` stay gitignored?** Guess: no. Un-ignore it so that phase 8 can add docs pages. `docs/gvf_discovery.md` is a session record and should not be committed as is.
11. **Is the `spinup.enabled: false` case meant to skip the event run** (finding 4)? Guess: no. The event should run from a cold start.
12. **Is `gvf` going to replace `modelled` as the default depth method?** That decides whether the round-0 code in rule 10 is CORE or soon STALE. Guess: not yet; keep both.
13. **Which `rp` name becomes the canonical one?** The rules say `rp_years`; the code mostly uses `rp_yr`. Guess: `rp_yr`, since it is already dominant and changes less.
14. **Should `tools/build_adaptation_lines.py` become a Snakemake rule?** Guess: yes. It is a per-basin step after rule 13.
15. **Should CI install the full env (micromamba from a Linux env file)?** Guess: yes, with a small `environment-ci.yml` for the unit tests.
16. **Should `Reference_memory.txt` be split into docs/ pages (methods, data, running), with history dropped and the section 5 domain facts kept?** Guess: yes, in phase 5.

**Decisions**

| Q | decision |
|---|---|
| 1, 2 | delete both archive folders |
| 3 | `adaptation_strategies_manual.yml` is dead: delete |
| 4 | move the analysis scripts to `analysis/`; delete the DUPLICATE and STALE ones |
| 5 | delete `KL_gtsm_storm_tide.py` and `check_surge_shape_duration.py` |
| 6 | reported upstream; runtime fix `src/hydromt_sfincs_patch.py` (done); delete the two root write-ups in phase 4 |
| 7, 8, 9 | delete the probe, prof and .tif, fix the `.gitignore` typo; untrack `figs/dag` and `qgis` |
| 10 | un-ignore `/docs/` |
| 11 | event runs without spin-up (done for `build`; `adapt` in TODO.md) |
| 12 | keep both depth methods |
| 13 | canonical return-period name is `rp_yr` (update the global rules' `rp_years` example) |
| 14 | make `tools/build_adaptation_lines.py` a Snakemake rule |
| 15 | CI pip list completed (done); cross-platform env in phase 8 |
| 16 | split Reference_memory.txt into docs/ pages, keeping the domain facts |

## 8. Plan (phases 3-8)

Baselines in `.baseline/` and CLAUDE.md:
- lint 745 at the baseline commit (717 after the `Leuven_et_al/` deletion)
- pytest 45 passed
- check_code_health OK
- dry runs for Ebro `default`: preprocess 22 jobs, build 25 jobs

| phase | scope | verification |
|---|---|---|
| 3. Smoke tests and quick-look helpers | Smoke tests on synthetic inputs for the untested CORE modules, first `raster`, `grid`, `io`, `extreme_values` (return level monotonic in RP), `surge`, `river_forcing`, `protection_weir`, `river_burn`, `river_conditioning` (non-increasing downstream) and `adaptation_scaling`. Fix the CI dependency set (finding 5). Add a `src/quicklook.py` with one panel per key variable for the raster/series outputs of rules 05, 07, 09, 10, 13 and 16, using fixed colour scales and stage/scenario/hash titles. | `pytest` green locally and in CI. The new test count is recorded in `.baseline/`. Reference quick-look figures for 2433835 are rendered from existing results into `.baseline/figs/`. |
| 4. Delete stale and duplicate files | After answers to Q1-Q10: archives; `adaptation_strategies_manual.yml`; the probe, prof and .tif; the bug-fix .py/.txt; `figs/dag`; `qgis`; the DUPLICATE tools and tests scripts; dead functions `clip_ocean_from_topo` and `reproject_to_reference_grid`; dead config keys (`gvf.target_discharge`, catalogue `river_basins`). Fix the `.gitignore` typo and the stale exclude lists. | Repo-wide `git grep` shows 0 references to each deleted name. pytest 45+ passed. check_code_health OK. Dry runs unchanged at 22/25 jobs. Lint count recorded. |
| 5. Documentation trim | Strip history from code comments and docstrings (finding 40). Fix the contradicting docstrings (finding 41) and the "rules 01-12" in Snakefile and CLAUDE.md. Distil Reference_memory.txt into docs/ pages, keeping the section 5 domain facts, then delete it. Shorten CHANGELOG (git is the changelog) or keep it as release notes only, per your call. | Lint only. The diff touches comments and docstrings only (AST of code bodies unchanged, checked per file). Report the domain facts kept, with their new location. |
| 6. Refactors, one module per session | In order of risk and payoff: (a) shared helpers for reach-ID normalisation, zsini dry-out, sfincs.inp writing and discharge-point placement, moved into src (findings 2, 7-9); (b) script 10 split into src functions plus a thin `main()` (finding 19); (c) one adaptation measure framework for pre and post (finding 11); (d) one topo-sort and one raster/polyline I/O helper (findings 13, 16); (e) MDT loading in 05a moved onto `src/surge` (finding 14); (f) path helpers in 00_common.smk and shared param blocks for 13/16/18a (findings 10, 18); (g) narrow the broad excepts (finding 1), print → logging, library I/O moved to the scripts. | Parity per refactor on 2433835 `default`: identical `snakemake -n` job lists. For the touched rule outputs, rerun only the affected rules on the smallest scenario and compare raster stats, DataFrame equality and output hashes against `.baseline/`. Quick-look figures match the reference. |
| 7. Consistency sweep | Report first, then fix on approval: one name per concept (RP, depth, dx, reach ID, sea mask, basin); rename misleading rule and file names (10 "modelled", 11 "testing", 09 file vs rule, the two 13s); nodata and magic constants into config; dead config branches (finding 29) and the spin-up-gated event (finding 4); docs vs code; lint `workflow/scripts/` in pre-commit and work the ruff count down. | Config materialized and compared before and after. Dry-run job counts unchanged (renamed rules mapped one to one). pytest green. The ruff count drops and the change is recorded. |
| 8. Portability and README | `site.example.yml` (or `.env.example`) for the GCFM_* variables and the SFINCS exe, failing loudly when missing; remove personal paths; a cross-platform env file; README rewritten as the front door (about 150 lines); docs/ pages for install, data (table generated from data_catalogue.yml with version, source, licence and how to register), running and methods; LICENSE and CITATION.cff. | A cold-start subagent follows only the README in a clean checkout and reports every gap. `snakemake -n` resolves with only the example-file values filled in. |
