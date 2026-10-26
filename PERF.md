# Performance baseline

## Applied optimizations

- O1 test_bifurcation_calibration_options: CRS from layer metadata (pyogrio.read_info) — median wall 259.4 s -> 3.0 s (-98.8%), peak RSS 3305 -> 250 MB, outputs bit-identical
- O2 get_river_network: CRS from layer metadata (pyogrio.read_info) — median wall 12.0 s -> 1.0 s (-91.7%), peak RSS 3309 -> 128 MB, features identical (columns, dtypes, row order, WKB, CRS); GPKG bytes differ only in gpkg_contents.last_change, which also differs between repeat runs of unchanged code
- O4 run_event: coarse->subgrid map built once, numpy per frame — median wall 1277.6 -> 572.0 s (-55.2%), peak RSS 2816 -> 2165 MB, flood_timeseries max abs diff 0 (area) / 0 (volume); flood_timeseries.csv, PNG and MP4 bit-identical; compute_flood_timeseries_stats alone 663.1 -> 39.9 s on the same sfincs_map.nc
- O7 modelled_depth_estimation: subgrid table reused when its inputs are unchanged (round 2 = round 1; tile-level reuse gains nothing on Ebro, whose 4 tiles all change round 0 -> 1) — median wall 274.5 -> 218.9 s (-20.3%), peak RSS 3123 -> 3206 MB (within run spread 3097-3288), subgrid max abs diff 0 (sfincs_subgrid.nc, dep/manning tifs byte-identical in all rounds; zs/zsmax, calibration_state.csv and declared outputs identical)

Wall time, memory, data volume and profiles for the smallest end-to-end case.
Raw measurements, profiles and the measurement tools are in `.baseline/perf/` (gitignored).

## 1. Setup

| item | value |
|---|---|
| scenario | basin `2433835` (Ebro delta), scenario `default`, target `all` (preprocess + build, 28 jobs incl. `all`) |
| commit | `6b05b4a` (clean tree), run from a scratch copy of the working tree |
| command | `snakemake all --cores 14 --drop-metadata --config target_basins="[2433835]" target_scenarios="['default']"` |
| cores / RAM | 14 logical / 12 physical, 15.4 GB; `--cores 14` |
| date | 2026-10-22 (pass 1 18:38-19:26, pass 2 19:28-20:17) |
| raw data | `D:\GCFM_UU\raw_data`, local but on an **external USB HDD** (WD My Passport); read only |
| results / scratch | fresh results dir on C: (NVMe SSD), deleted after the run |
| env | `hmt_sfincs_dev` (Python 3.11, snakemake 9.21.0), SFINCS v2.3.0 `sfincs.exe` from `config/site.yml` |

`--drop-metadata` was needed because the scratch location makes Snakemake's
`.snakemake/metadata/<base64(output path)>` name exactly 260 characters for
`10_calibration/dike_positions_wetland_switch.png`, and `LongPathsEnabled=0` on this
machine. A first attempt without the flag stopped after `modelled_depth_estimation`
with `WorkflowError: Error recording metadata for finished job`. The job script itself
had succeeded. The repo's normal `results_dir` (`D:\GCFM_UU\results`) gives shorter paths.

Method. Pass 1 ran once with no profiler. A psutil monitor launched Snakemake and sampled
the whole process tree every 0.5 s. A process was attributed to a job via the
`.snakemake/scripts/tmp*.<script>.py` in its command line, and child processes
(`sfincs.exe`, multiprocessing workers) inherit the job of their parent. Wall times come
from the Snakemake log; the monitor agrees within 1 s for every job. Pass 2 re-ran the
top 5 stages one at a time (`--forcerun <rule> --allowed-rules <rule>`, each checked with
`-n` to be exactly 1 job). It used cProfile and an I/O-call wrapper loaded through a
`sitecustomize.py` on `PYTHONPATH`. Spawned worker processes are profiled as well.

## 2. All jobs (pass 1)

Rows are sorted by wall time. Peak RSS is the maximum over samples of the summed RSS of the
job's process tree. Input/output GB are the declared Snakemake inputs and outputs, with
directories summed. Read/write GB are psutil `io_counters` of the job's process tree.
"CPU cores" is CPU time divided by wall time.

| stage (rule) | wall s | peak RSS MB | share of total | input GB | output GB | read GB (io) | write GB (io) | throughput (input GB / wall min) | CPU cores (avg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run_event | 1277 | 2802 | 35.0% | 0.002 | 0.102 | 6.291 | 3.572 | 0.000 | 3.02 |
| get_river_network | 795 | 3149 | 21.8% | 1.318 | 0.000 | 1.935 | 0.000 | 0.099 | 0.03 |
| extract_gtsm_series | 681 | 528 | 18.7% | 0.000 ² | 0.037 | 39.129 | 0.095 | 0.000 | 0.36 |
| modelled_depth_estimation | 275 | 3127 | 7.5% | 0.278 | 0.002 | 1.869 | 0.506 | 0.061 | 2.44 |
| test_bifurcation_calibration_options | 260 | 3570 | 7.1% | 2.633 | 0.000 | 3.884 | 0.000 | 0.608 | 0.17 |
| get_boundary_forcings | 84 | 533 | 2.3% | 123.545 ¹ | 0.023 | 1.670 | 0.023 | 88.247 | 0.50 |
| build_sfincs_skeleton | 74 | 1858 | 2.0% | 0.014 | 0.050 | 0.210 | 0.086 | 0.011 | 0.97 |
| get_elevation | 35 | 977 | 1.0% | 263.974 ¹ | 0.003 | 0.442 | 0.014 | 452.527 | 0.39 |
| run_spinup | 32 | 2268 | 0.9% | 0.003 | 0.004 | 1.130 | 0.004 | 0.005 | 4.75 |
| storm_tide_hydrographs | 28 | 886 | 0.8% | 0.040 | 0.002 | 0.150 | 0.002 | 0.086 | 0.90 |
| sanity_checks | 17 | 1911 | 0.5% | 0.004 | 0.000 | 0.413 | 0.080 | 0.015 | 0.95 |
| build_sfincs | 12 | 461 | 0.3% | 0.003 | 0.000 | 1.034 | 0.000 | 0.015 | 0.80 |
| compute_flood_metrics | 12 | 1881 | 0.3% | 0.102 | 0.054 | 0.192 | 0.054 | 0.508 | 0.89 |
| prepare_landuse | 9 | 386 | 0.2% | 1.715 | 0.001 | 0.041 | 0.000 | 11.432 | 0.34 |
| get_roughness | 8 | 581 | 0.2% | 0.005 | 0.007 | 0.071 | 0.007 | 0.034 | 0.82 |
| determine_model_domain | 7 | 301 | 0.2% | 0.000 | 0.000 | 0.069 | 0.000 | 0.001 | 0.56 |
| plot_river_network | 6 | 214 | 0.2% | 0.000 | 0.000 | 0.063 | 0.000 | 0.005 | 0.47 |
| grid_align_landuse | 6 | 555 | 0.2% | 0.001 | 0.001 | 0.083 | 0.001 | 0.013 | 0.76 |
| get_protection_levels | 5 | 241 | 0.1% | 1.732 | 0.000 | 0.093 | 0.000 | 20.778 | 0.64 |
| get_landuse | 5 | 385 | 0.1% | 0.005 | 0.008 | 0.093 | 0.008 | 0.055 | 0.72 |
| clean_river_network | 5 | 260 | 0.1% | 0.001 | 0.000 | 0.101 | 0.001 | 0.010 | 0.79 |
| select_surge_stations | 3 | 176 | 0.1% | 0.003 | 0.000 | 0.040 | 0.000 | 0.058 | 0.69 |
| test_upstream_boundary | 3 | 220 | 0.1% | 0.002 | 0.000 | 0.053 | 0.000 | 0.045 | 0.57 |
| split_delta_polygons | 2 | 116 | 0.1% | 0.000 | 0.000 | 0.022 | 0.000 | 0.003 | 0.57 |
| get_land_polygons | 2 | 139 | 0.1% | 0.001 | 0.000 | 0.030 | 0.000 | 0.036 | 0.62 |
| enforce_river_monotonicity | 2 | 162 | 0.1% | 0.003 | 0.003 | 0.036 | 0.000 | 0.098 | 0.50 |
| build_sfincs_grid | 1 | 104 | 0.0% | 0.000 | 0.000 | 0.020 | 0.000 | 0.006 | 0.58 |
| **sum of jobs** | **3646** | | 100% | 395.4 | 0.298 | 59.2 | 4.45 | | |
| **end-to-end** | **2847** | 3872 ³ | | | | | | | |

¹ The declared input includes whole global dataset directories (`FathomDEM` 256.1 GB, `GloFAS` 113.7 GB,
`gebco_2026_geotiff` 7.5 GB, SLR `confidence_output_files` 9.6 GB). Compare with the read column for what was actually read.
² The GTSM inputs are `params`, not declared inputs: `GTSM_storm_tide_hourly/{total_water_level,storm_surge_residual}`,
1800 monthly files, 65.0 GB on disk. The job read 39.1 GB of it.
³ Peak RSS summed over all concurrent jobs (at t = 1180 s, during `modelled_depth_estimation`).

Totals: the jobs sum to 3646 s against **2847 s end to end** (47.5 min), an overlap factor of 1.28.
Nearly all of the overlap is in rules 01-09, which share the first ~900 s:

- `get_river_network` (9-804 s) and `extract_gtsm_series` (12-693 s) ran concurrently and both read from the same USB HDD.
- `get_boundary_forcings`, `get_elevation`, `plot_river_network`, `get_landuse` and `get_roughness` started together at 816 s.
- `test_bifurcation_calibration_options` (900-1160 s) overlapped only rules 08-09 (7 s).

From rule 10 onward the chain is serial: `modelled_depth_estimation` → `build_sfincs_skeleton` → `run_spinup` →
`build_sfincs`/`sanity_checks` → `run_event` → `compute_flood_metrics`, 1160-2847 s. That is 1687 s, 59% of end to end.
The four SFINCS-running jobs run `sfincs.exe` for 572 s in total: rule 10 has 3 rounds of 18-21 s, spin-up 19 s and the event 495 s.

## 3. Top 5 stages (pass 2, profiled)

The profiled re-runs took:

| stage | pass 2 wall s | pass 1 wall s |
|---|---:|---:|
| `run_event` | 1287 | 1277 |
| `get_river_network` | 125 | 795 |
| `extract_gtsm_series` | 523 | 681 |
| `modelled_depth_estimation` | 304 | 275 |
| `test_bifurcation_calibration_options` | 618 | 260 |

The large differences in the three D:-bound stages come from HDD contention in pass 1
(concurrent jobs on the same disk) and from the OS file-cache state, not from the profiler.
Shares below are relative to the pass-2 wall time of the stage.

Function tables show project functions (`workflow/src`, `workflow/scripts`) plus library
functions called directly from project code ("under" names the project caller). Line
numbers in `workflow/scripts/*` are corrected for Snakemake's 3-line preamble. Cumulative
times nest, so the shares do not add up to 100%. "wrapped I/O API call" is the profiling
wrapper around `geopandas.read_file` / `netCDF4.Dataset` etc.; the API and file are in the
I/O inventory.

### 3.1 run_event (rule 16): 1287 s, peak 2844 MB

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| <script body> | workflow/scripts/16_run_event.py:1 | 1286.2 | 1 | 99.9% |
| compute_flood_timeseries_stats | workflow/src/postprocessing.py:259 | 732.4 | 1 | 56.9% |
| run_sfincs_subprocess | workflow/src/sfincs_run.py:305 | 488.4 | 1 | 37.9% |
| wait (under run_sfincs_subprocess) | stdlib/subprocess.py:1259 | 488.4 | 1 | 37.9% |
| downscale_floodmap (under compute_flood_timeseries_stats, compute_max_inundation) | hydromt_sfincs/utils.py:1253 | 406.6 | 338 | 31.6% |
| compute (under compute_flood_timeseries_stats, compute_max_inundation) | xarray/core/dataarray.py:1216 | 249.9 | 341 | 19.4% |
| where (under compute_flood_progression, compute_flood_timeseries_stats, compute_max_inundation) | xarray/core/common.py:1131 | 53.0 | 340 | 4.1% |
| animate_flood_progression | workflow/src/plots.py:2297 | 49.9 | 1 | 3.9% |
| save (under animate_flood_progression) | matplotlib/animation.py:943 | 46.5 | 1 | 3.6% |
| sum (under compute_flood_timeseries_stats, inundation_stats) | xarray/core/_aggregations.py:3140 | 17.4 | 676 | 1.4% |
| _find_and_load (under <module>, animate_flood_progression) | <frozen importlib._bootstrap>:1165 | 14.3 | 20 | 1.1% |
| compute_max_inundation | workflow/src/postprocessing.py:156 | 6.2 | 1 | 0.5% |
| <module> | workflow/src/postprocessing.py:1 | 4.7 | 1 | 0.4% |
| reproject_max_for_plot | workflow/src/plots.py:1639 | 3.4 | 2 | 0.3% |
| wrapper (under reproject_max_for_plot) | rasterio/env.py:407 | 2.7 | 340 | 0.2% |

`sfincs.exe`: one run, 488 s = **37.9%** of the stage (3065 CPU-s in pass 1, ~6.2 cores while it runs).
The remaining 57% is post-processing. `compute_flood_timeseries_stats` loops over 338 output frames, calling `hydromt_sfincs.utils.downscale_floodmap` and an xarray/dask `compute` on each.
Self time: 462 s in `_thread.lock.acquire` (the main thread waiting on dask worker threads, which cProfile does not see), 96 s in `affine.__mul__` (6588 calls, coordinate arrays), 34 s in `sha1` (dask tokenizing).
Project code accounts for 4.7 s of self time. Outside the solver the job runs at about 1 core: 790 CPU-s over about 790 s.

I/O inventory (the ~6.3 GB of io reads are repeated reads of the files below):

| file | size MB | times opened (API) | access |
|---|---:|---|---|
| <results>/2433835/runs/default/sfincs/sfincs_map.nc | 96.1 | 3 (xarray.backends.api.open_dataset) | chunked |
| <results>/2433835/sfincs_skeleton/subgrid/dep_subgrid.tif | 24.0 | 2 (rioxarray.open_rasterio) | windowed reads x339; chunked |
| <results>/2433835/preprocessing_inputs/domain/2433835_zsini_sea_cells_on_grid.tif | 0.5 | 3 (rioxarray.open_rasterio) | windowed reads x5; chunked |
| <results>/2433835/sfincs_skeleton/sfincs.ind | 0.3 | 3 (builtins.open) | full |
| <results>/2433835/sfincs_skeleton/sfincs.dep | 0.3 | 3 (builtins.open) | full |
| <results>/2433835/sfincs_skeleton/sfincs.manning | 0.3 | 3 (builtins.open) | full |
| <results>/2433835/sfincs_skeleton/sfincs.ini | 0.3 | 3 (builtins.open) | full |
| <results>/2433835/preprocessing_inputs/domain/2433835_land_mask_on_grid.gpkg | 0.1 | 2 (geopandas.read_file) | filtered:bbox |
| <results>/2433835/preprocessing_inputs/domain/2433835_river_network_clean.gpkg | 0.1 | 2 (geopandas.read_file) | full |
| <results>/2433835/preprocessing_inputs/domain/2433835_domain.gpkg | 0.1 | 1 (geopandas.read_file) | full |
| <results>/2433835/runs/default/visuals/01_inundation_ratio.png | 0.1 | 1 (builtins.open) | write |

### 3.2 get_river_network (rule 06): 125 s isolated / 795 s in pass 1, peak 3204 MB

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| <script body> | workflow/scripts/06_get_river_network.py:1 | 124.8 | 1 | 99.7% |
| wrapped I/O API call (see I/O inventory) (under <module>, load_domain) | - | 123.2 | 4 | 98.4% |
| _find_and_load (under <module>) | <frozen importlib._bootstrap>:1165 | 0.8 | 6 | 0.7% |
| <built-in method _pickle.loads> (under <module>) | ~:0 | 0.2 | 1 | 0.1% |
| to_file (under <module>) | geopandas/geodataframe.py:1530 | 0.1 | 1 | 0.1% |
| load_domain | workflow/src/domain.py:39 | 0.1 | 1 | 0.0% |
| <module> | workflow/src/profiling.py:1 | 0.0 | 1 | 0.0% |
| clip (under <module>) | geopandas/tools/clip.py:109 | 0.0 | 1 | 0.0% |
| bounds_to_wgs84 | workflow/src/domain.py:13 | 0.0 | 1 | 0.0% |
| from_string (under bounds_to_wgs84) | pyproj/crs/crs.py:436 | 0.0 | 1 | 0.0% |
| __init__ (under <module>) | geopandas/geodataframe.py:150 | 0.0 | 1 | 0.0% |
| to_crs (under load_domain) | geopandas/geodataframe.py:1757 | 0.0 | 1 | 0.0% |
| info (under <module>) | stdlib/logging\__init__.py:1479 | 0.0 | 4 | 0.0% |
| crs (under <module>) | geopandas/geodataframe.py:557 | 0.0 | 5 | 0.0% |
| from_crs (under bounds_to_wgs84) | pyproj/transformer.py:551 | 0.0 | 1 | 0.0% |

`sfincs.exe`: none.
The wrapped call is `geopandas.read_file` → `pyogrio.read`. The I/O timeline splits it into two reads of the 1.26 GB global SWORD GeoPackage:
- the CRS probe `gpd.read_file(..., rows=0)` at `workflow/scripts/06_get_river_network.py:32` takes **123.5 s** (pyogrio receives `max_features=0`);
- the bbox read on the next line takes 0.3 s.
The job read 1.94 GB with 19 CPU-s (0.15 cores). In pass 1 the same read took 795 s at 0.03 cores while `extract_gtsm_series` read 39 GB from the same USB HDD.

| file | size MB | times opened (API) | access |
|---|---:|---|---|
| <raw>/SWORD/SWORD_global_v17c_unpublished_modified.gpkg | 1256.9 | 2 (geopandas.read_file) | filtered:bbox; filtered:max_features; filtered:rows |
| <results>/2433835/preprocessing_inputs/domain/2433835_domain.gpkg | 0.1 | 1 (geopandas.read_file) | full |
| <results>/2433835/preprocessing_inputs/domain/domain_bbox.json | 0.0 | 1 (builtins.open) | full |

### 3.3 extract_gtsm_series (rule 07b): 523 s isolated / 681 s in pass 1, peak 536 MB

Main process:

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| <script body> | workflow/scripts/07b_extract_gtsm_series.py:1 | 523.1 | 1 | 100.0% |
| <dictcomp> | workflow/scripts/07b_extract_gtsm_series.py:35 | 517.9 | 1 | 98.9% |
| extract_station_series | workflow/src/gtsm.py:112 | 517.9 | 2 | 98.9% |
| _chain_from_iterable_of_lists (under extract_station_series) | stdlib/concurrent\futures\process.py:614 | 517.1 | 1802 | 98.8% |
| to_netcdf (under <module>) | xarray/core/dataset.py:2039 | 2.2 | 1 | 0.4% |
| _find_and_load (under <module>, station_coordinates) | <frozen importlib._bootstrap>:1165 | 1.1 | 6 | 0.2% |
| __init__ (under <module>) | xarray/core/dataset.py:378 | 1.1 | 1 | 0.2% |
| station_coordinates | workflow/src/gtsm.py:36 | 0.9 | 1 | 0.2% |
| monthly_files | workflow/src/gtsm.py:25 | 0.6 | 3 | 0.1% |
| <built-in method builtins.sorted> (under monthly_files) | ~:0 | 0.6 | 3 | 0.1% |
| __exit__ (under extract_station_series) | stdlib/concurrent\futures\_base.py:646 | 0.4 | 2 | 0.1% |
| <built-in method _pickle.loads> (under <module>) | ~:0 | 0.2 | 1 | 0.0% |
| wrapped I/O API call (see I/O inventory) (under station_coordinates) | - | 0.1 | 1 | 0.0% |
| map (under extract_station_series) | stdlib/concurrent\futures\process.py:813 | 0.1 | 2 | 0.0% |
| duplicated (under extract_station_series) | pandas/core/indexes/base.py:3129 | 0.1 | 2 | 0.0% |

Worker processes (4 spawned `ProcessPoolExecutor` workers, 2 at a time per variable, profiles summed; shares exceed 100% because workers run in parallel):

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| _read_month | workflow/src/gtsm.py:80 | 1026.5 | 1800 | 196.1% |
| wrapped I/O API call (see I/O inventory) (under _read_month) | - | 535.2 | 1800 | 102.3% |
| _find_and_load (under <module>, _read_month) | <frozen importlib._bootstrap>:1165 | 4.9 | 20 | 0.9% |
| <script body> | workflow/scripts/07b_extract_gtsm_series.py:1 | 4.8 | 4 | 0.9% |
| <built-in method numpy.asarray> (under _read_month) | ~:0 | 1.4 | 1800 | 0.3% |
| __array_wrap__ (under _read_month) | numpy/ma/core.py:3133 | 0.4 | 8352 | 0.1% |
| __new__ (under _read_month) | numpy/ma/core.py:2872 | 0.3 | 8208 | 0.1% |
| _StartCountStride (under _read_month) | netCDF4/utils.py:81 | 0.2 | 3600 | 0.0% |
| __mul__ (under _read_month) | numpy/ma/core.py:4339 | 0.2 | 2664 | 0.0% |
| where (under _read_month) | numpy/ma/core.py:7913 | 0.1 | 1872 | 0.0% |
| _safecast (under _read_month) | netCDF4/utils.py:15 | 0.1 | 2736 | 0.0% |
| all (under _read_month) | numpy/ma/core.py:4976 | 0.1 | 1800 | 0.0% |
| __rmul__ (under _read_month) | numpy/ma/core.py:4345 | 0.1 | 936 | 0.0% |
| argsort (under _read_month) | numpy/_core/fromnumeric.py:1100 | 0.1 | 5400 | 0.0% |
| __eq__ (under _read_month) | numpy/ma/core.py:4266 | 0.1 | 1872 | 0.0% |

`sfincs.exe`: none.
Summed over workers: 535 s in `netCDF4.Dataset` opens (1800 files) and 488 s in the per-station hyperslab reads inside `_read_month` (netCDF4 C code, attributed to the caller).
The workers read 39.1 GB in 523 s (**75 MB/s**) with 250 CPU-s in total, about 0.24 cores per worker. The main process spends 99% of its time waiting on the pool.

| file | size MB | times opened (API) | access |
|---|---:|---|---|
| <raw>/GTSM_storm_tide_hourly/total_water_level/ (900 files) | 33744.3 | 900 (netCDF4.Dataset) | partial: header + `time` + station-column hyperslab per file |
| <raw>/GTSM_storm_tide_hourly/storm_surge_residual/ (900 files) | 28288.5 | 901 (netCDF4.Dataset) | partial: header + `time` + station-column hyperslab per file |
| <results>/2433835/preprocessing_inputs/forcing/gtsm_hourly.nc | 35.6 | 1 (netCDF4.Dataset) | write |

Each monthly file is opened once; only the matched station columns of the `(time, station)` variable are sliced, plus `time`.

### 3.4 modelled_depth_estimation (rule 10): 304 s, peak 3152 MB

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| <script body> | workflow/scripts/10_depth_estimation_modelled.py:1 | 301.9 | 1 | 99.5% |
| _create_subgrid | workflow/scripts/10_depth_estimation_modelled.py:788 | 178.1 | 3 | 58.7% |
| create (under _create_subgrid) | hydromt_sfincs/components/grid/subgrid.py:517 | 178.1 | 3 | 58.7% |
| _run_calibration_round | workflow/scripts/10_depth_estimation_modelled.py:708 | 70.8 | 3 | 23.3% |
| run_sfincs_subprocess | workflow/src/sfincs_run.py:305 | 57.7 | 3 | 19.0% |
| wait (under run_sfincs_subprocess) | stdlib/subprocess.py:1259 | 57.7 | 3 | 19.0% |
| _round_diagnostics | workflow/scripts/10_depth_estimation_modelled.py:833 | 34.6 | 3 | 11.4% |
| compute_max_inundation | workflow/src/postprocessing.py:156 | 15.2 | 3 | 5.0% |
| animate_flood_progression | workflow/src/plots.py:2297 | 13.1 | 3 | 4.3% |
| save (under animate_flood_progression) | matplotlib/animation.py:943 | 11.9 | 3 | 3.9% |
| _find_and_load (under <module>, animate_flood_progression) | <frozen importlib._bootstrap>:1165 | 10.0 | 29 | 3.3% |
| write (under _run_calibration_round) | hydromt_sfincs/sfincs.py:328 | 8.8 | 3 | 2.9% |
| compute (under compute_max_inundation) | xarray/core/dataarray.py:1216 | 4.8 | 9 | 1.6% |
| write_netcdf | workflow/src/hydromt_sfincs_patch.py:56 | 4.1 | 3 | 1.3% |
| write_netcdf (under write_netcdf) | hydromt_sfincs/components/grid/subgrid.py:382 | 4.1 | 3 | 1.3% |

`sfincs.exe`: 3 calibration rounds of 17.5 / 20.0 / 20.0 s = 57.6 s, **19.0%** of the stage (447 CPU-s, ~7.8 cores while running).
Subgrid creation (`hydromt_sfincs` `SubgridTableRegular.create`, 3×) takes 59%. Its self time is mostly `scipy.interpolate` `griddata` `__call__` (91.5 s) and `process_tile_regular` (28.6 s).
Project code accounts for 0.9 s of self time. Outside the solver the job runs at about 1 core: 242 CPU-s over about 246 s.

| file | size MB | times opened (API) | access |
|---|---:|---|---|
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round0/sfincs_subgrid.nc | 46.4 | 1 (netCDF4.Dataset) | write |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round1/sfincs_subgrid.nc | 46.4 | 1 (netCDF4.Dataset) | write |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round2/sfincs_subgrid.nc | 46.4 | 1 (netCDF4.Dataset) | write |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round1/subgrid/dep_subgrid.tif | 24.0 | 1 (rioxarray.open_rasterio) | windowed reads x2; chunked |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round2/subgrid/dep_subgrid.tif | 24.0 | 1 (rioxarray.open_rasterio) | windowed reads x2; chunked |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round0/subgrid/dep_subgrid.tif | 23.9 | 1 (rioxarray.open_rasterio) | windowed reads x2; chunked |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round0/subgrid/manning_subgrid.tif | 7.4 | 6 (rasterio.open) | write |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round1/subgrid/manning_subgrid.tif | 7.4 | 6 (rasterio.open) | write |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round2/subgrid/manning_subgrid.tif | 7.4 | 6 (rasterio.open) | write |
| <results>/2433835/preprocessing_inputs/domain/2433835_roughness.tif | 6.2 | 3 (rioxarray.open_rasterio) | windowed reads x12; chunked |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round0/sfincs_map.nc | 3.1 | 8 (xarray.backends.api.open_dataset) | chunked; full(lazy) |
| <results>/2433835/preprocessing_inputs/domain/2433835_elevation_conditioned.tif | 3.0 | 3 (rioxarray.open_rasterio) | full reads x1, windowed reads x12; chunked |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round2/sfincs_map.nc | 2.9 | 6 (xarray.backends.api.open_dataset) | chunked; full(lazy) |
| <results>/2433835/preprocessing_inputs/depth_crest_calibration/round1/sfincs_map.nc | 2.9 | 6 (xarray.backends.api.open_dataset) | chunked; full(lazy) |

### 3.5 test_bifurcation_calibration_options (rule 11b): 618 s isolated / 260 s in pass 1, peak 3302 MB

| function | file:line | cumtime s | ncalls | share of stage |
|---|---|---:|---:|---:|
| <script body> | workflow/scripts/11b_bifurcation_calibration_options.py:1 | 617.3 | 1 | 99.9% |
| clip_and_preclean | workflow/scripts/11b_bifurcation_calibration_options.py:67 | 613.1 | 2 | 99.2% |
| wrapped I/O API call (see I/O inventory) (under <module>, clip_and_preclean, load_domain) | - | 612.4 | 9 | 99.1% |
| _find_and_load (under <module>) | <frozen importlib._bootstrap>:1165 | 7.0 | 24 | 1.1% |
| <module> | workflow/src/plots.py:1 | 1.5 | 1 | 0.2% |
| <module> | workflow/src/river_forcing.py:1 | 0.8 | 1 | 0.1% |
| <module> | workflow/src/extreme_values.py:1 | 0.8 | 1 | 0.1% |
| <built-in method _pickle.loads> (under <module>) | ~:0 | 0.2 | 1 | 0.0% |
| load_forcing_crossings | workflow/src/river_forcing.py:805 | 0.1 | 1 | 0.0% |
| <module> | workflow/src/profiling.py:1 | 0.1 | 1 | 0.0% |
| load_domain | workflow/src/domain.py:39 | 0.1 | 1 | 0.0% |
| clip (under clip_and_preclean) | geopandas/tools/clip.py:109 | 0.0 | 2 | 0.0% |
| fix_tjunction_tails | workflow/src/river_network.py:163 | 0.0 | 2 | 0.0% |
| to_crs (under <module>, _prep, fix_tjunction_tails) | geopandas/geodataframe.py:1757 | 0.0 | 9 | 0.0% |
| __init__ (under clip_and_preclean, identify_delta_outflow_points, load_forcing_crossings) | geopandas/geodataframe.py:150 | 0.0 | 5 | 0.0% |

`sfincs.exe`: none.
The wrapped calls are again `geopandas.read_file` → `pyogrio.read` on the two global SWORD GeoPackages. In `clip_and_preclean`, the CRS probe `gpd.read_file(global_path, rows=0)` at `workflow/scripts/11b_bifurcation_calibration_options.py:79` takes:
- **482.5 s** on `SWORD_global_v17c_unpublished.gpkg`;
- **130.4 s** on `SWORD_global_v17c_unpublished_modified.gpkg`;
- 0.2 s for each bbox read that follows.
The job read 3.88 GB with 48 CPU-s (0.08 cores). The 260 s in pass 1 is consistent with the files being partly in the OS file cache after rule 06 and other reads.

| file | size MB | times opened (API) | access |
|---|---:|---|---|
| <raw>/SWORD/SWORD_global_v17c_unpublished_modified.gpkg | 1256.9 | 2 (geopandas.read_file) | filtered:bbox; filtered:max_features; filtered:rows |
| <raw>/SWORD/SWORD_global_v17c_unpublished.gpkg | 1253.4 | 2 (geopandas.read_file) | filtered:bbox; filtered:max_features; filtered:rows |
| <results>/2433835/preprocessing_inputs/domain/2433835_land_polygons.gpkg | 0.2 | 1 (geopandas.read_file) | full |
| <results>/2433835/preprocessing_inputs/forcing/river_forcing.nc | 0.2 | 1 (xarray.backends.api.open_dataset) | full(lazy) |
| <results>/2433835/preprocessing_inputs/domain/2433835_domain.gpkg | 0.1 | 1 (geopandas.read_file) | full |
| <results>/2433835/preprocessing_inputs/domain/2433835_delta_polygon.gpkg | 0.1 | 1 (geopandas.read_file) | full |
| <results>/2433835/preprocessing_inputs/domain/domain_bbox.json | 0.0 | 1 (builtins.open) | full |

## 4. Verdicts

| stage | verdict | evidence |
|---|---|---|
| `run_event` | CPU-bound in a library | Of 1287 s: 57% in `hydromt_sfincs.downscale_floodmap` + xarray/dask over 338 frames at ~1 core (project self time 4.7 s); `sfincs.exe` takes 38%. |
| `get_river_network` | I/O-bound | 123 of 125 s in one pyogrio read (`rows=0` CRS probe, line 32) at 0.15 cores, reading 1.9 GB from the USB HDD; 795 s when sharing the disk in pass 1. |
| `extract_gtsm_series` | I/O-bound | 39.1 GB from 1800 netCDF files at 75 MB/s; workers at ~0.24 cores each. |
| `modelled_depth_estimation` | CPU-bound in a library | 59% in hydromt_sfincs subgrid `create` (scipy `griddata` 91.5 s self) at ~1 core; `sfincs.exe` takes 19%. |
| `test_bifurcation_calibration_options` | I/O-bound | 612 of 618 s in two pyogrio `rows=0` CRS probes (line 79) on 1.3 GB GeoPackages at 0.08 cores. |

No stage is memory-bound: the largest job peak is 3.6 GB and the largest concurrent peak 3.9 GB, on 15.4 GB RAM.

## 5. Measurement limits

- Single run per pass, no repetitions. Stages that read from D: varied by up to 6× between passes (get_river_network 795 vs 125 s, test_bifurcation 260 vs 618 s). The causes are disk contention and the OS file cache, which was not flushed between passes.
- Snakemake log timestamps have 1 s resolution. The monitor samples every 0.5 s, so processes living less than 0.5 s are missed and the final CPU/io counters of a process can lag by up to 0.5 s.
- Windows `io_counters` count all read/write calls of the process, including cache hits and pipes. Memory-mapped and paging I/O are excluded. They are not physical disk bytes.
- cProfile sees only the main thread of each process. Dask/numba worker-thread time appears as `lock.acquire` in the caller, and time in C extensions (netCDF4, pyogrio, GDAL) is attributed to the calling Python frame. `sfincs.exe` time is measured from the process tree (0.5 s resolution) and shows in the profile as `subprocess.wait`. Profiling overhead was 1-10% for CPU-bound stages (run_event 1287 vs 1277 s, rule 10 304 vs 275 s).
- The I/O wrapper counts Python-level open/read calls. GDAL/HDF5 internal block reads are not itemised; "windowed/full" refers to `DatasetReader.read` calls, and "chunked" to xarray/rioxarray opens with `chunks`.
- `--drop-metadata` (see Setup) only skips Snakemake's provenance records after each job.

## Reproduce

Tools are in `.baseline/perf/tools/`. `env.sh`, `pass1.sh` and `pass2.sh` hold the scratch paths used here.

1. Copy `git ls-files -co --exclude-standard` plus `config/site.yml` to a scratch directory.
2. Set `GCFM_RESULTS_DIR` to an empty scratch directory and `GCFM_RAW_DATA_ROOT` to the raw data root.
3. From the scratch copy, run:

```
python .baseline/perf/tools/monitor.py .baseline/perf/tools/rule_scripts.txt pass1_samples.jsonl \
  snakemake all --cores 14 --drop-metadata --config target_basins="[2433835]" target_scenarios="['default']"
python .baseline/perf/tools/analyze_pass1.py pass1_samples.jsonl .snakemake/log/<latest>.snakemake.log pass1.json
```

Pass 2: put `.baseline/perf/tools/` (which holds `sitecustomize.py`) on `PYTHONPATH`, set `GCFM_PERF_OUT` and `GCFM_PERF_RULEMAP`, then per rule run
`snakemake <output> --forcerun <rule> --allowed-rules <rule> --cores 14 --drop-metadata --config ...`.
Collapse the profiles with `collapse_prof.py <rule>.prof <rule>_wall.json <stage_wall_s> 15`.
