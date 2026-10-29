# TODO

Decision table; mark each row's decision. Previous table: git show 580633b:TODO.md

## Behaviour

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|
| B26 | workflow/scripts/11_depth_estimation_modelled.py:1092 | calibration rounds have single unstable cells swinging metres per hour (2433835: (372,212) at the domain edge beside open-boundary cells; river cell (263,57)); with every crest 1 m above any water, round 2 still differs from round 1 by up to 8.6 cm (200 cells >1 cm, avg dt 4.374 -> 4.369 s) | behaviour | M | crests sit within model noise; round-2 exceedances | accepted as minor (documented in docs/methods.md); revisit only if production runs show the same cells |
| B25 | workflow/src/river_forcing.py:156 | GloFAS cell = highest-mean cell within 5 km of each crossing; no check that two crossings take the same or a main-stem cell (none shared in the current 8 basins) | behaviour | S | double-counted inflow if it happens | decide: fail or warn when cells are shared, and check cell mean against SWORD facc |
| B22 | workflow/src/surge.py:635 | rule select_surge_stations prints shapely 'invalid value encountered in distance' for every basin; COAST-RP coordinates have no NaN, so a NaN/empty geometry on the domain side is the likely source; stations are still found (Ebro: 9 within 10 km) | behaviour | S | none seen | decide: find the NaN geometry and drop it explicitly |
| B28 | workflow/src/surge.py:635 | select_surge_stations ranks stations by unsigned distance to the domain outline, so a station inside a bay ranks like one offshore and its locally amplified level forces the open boundary (analysis/check_surge_boundary.py: 4267691 has 8 of 10 stations inside, RP250 0.9-4.6 m; 1248635 0.92 m spread) | behaviour | M | coastal forcing level | decide: prefer stations outside or near the water-level boundary, or drop level outliers |
| B29 | workflow/scripts/14_build_sfincs.py:153 | comment says stations are matched to the nearest boundary cell; hydromt keeps them at their own coordinates and SFINCS interpolates each boundary cell from its two nearest | text | S | none | fix comment |
| B30 | <results_dir>/{1248635,1475442,2444235,4267691,620947,2209166}/sfincs_skeleton | skeletons are older than landuse_on_grid.tif (different grid); 2209166's default run no longer matches its skeleton | behaviour | M | results stale vs current preprocessing | rerun build for these basins |
| B31 | analysis/benchmark_upstream_boundary.py:1 | stress-case (RP100 + 1 m SLR) SFINCS benchmark missing for 3279946 (run stopped after 2 h 18 min); 2433835 and 2209166 pass; figs/upstream_boundary_benchmark/, the margin_km comment in config/config.yml, docs/methods.md, CHANGELOG.md and HANDOFF_upstream_boundary.md still give RP100 + 0.5 m numbers | behaviour | M | domain-size check incomplete for one basin | rerun `benchmark_upstream_boundary.py 3279946 --days 8 --threads 8 --cases swl_ext` (>2.5 h), then `2433835 2209166 3279946 --skip-run`; rerun check_upstream_boundary.py for 2209166, 3279946; confirm 5 km margin; update numbers |
| B32 | analysis/check_event_timing.py:1 | the SFINCS river peak lag along the path cannot be read from the compound `default` runs (the surge sets the coastal water-level peak) and no river-only run exists | behaviour | M | analysis/estimate_peak_travel.py unvalidated; on 3279946 its regime and kinematic celerities disagree on whether the lag is material (7 h vs 35 h, tau95 15.5 h) | run a river-only scenario at the default river RP for 2433835 and 3279946, then add the obs-point lag to check_event_timing.py and compare |
| B33 | workflow/src/river_bed.py:320 | on 3279946 the new zsini river level (steady GVF at Q_mean on rule 11's bed) lies below the old spin-up's end level (median -0.67 m, p5 -2.72 m over 1987 river nodes); 2433835 and 2209166 match within 0.3/0.4 m. That spin-up was still falling 1-5 cm per 3 h at its end, and GVF is 1D in-channel | behaviour | M | spin-up of 3279946 now fills instead of draining (smaller transient than the old median +1.51 m) | after the rebuild, compare mean_level_m with the new spin-up end; add rule 11's steady_check to rule 13a |
| B27 | workflow/src/grid.py:49 | hydromt `create_grid_from_region(align=True)` rounds the origin, so the axis-aligned grid can miss up to half a cell of the delta polygon at each edge (synthetic test: 80 m at dx 500 m) | behaviour | S | domain edge cells | decide: floor/ceil the bounds to dx before fitting |

## Structure

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|
| S3 | workflow/src/water_surface.py:73 | redefines LANDUSE_SEA_CODE = 200 instead of importing it from src.landuse (two sources of truth for the sea code) | structure | S | none | import from src.landuse |
| S2 | workflow/src/river_bed.py:1 | validate the round-0 river bed against surveyed thalweg bathymetry (Mississippi, Rhine-Meuse): pass if median \|Δz\| ≤ 2 m with no systematic sign over the backwater reach; if it fails, the fused slope estimator of `<raw_data_root>/river_surface_slope_plan.md` §3 is the fallback | structure | S | river bed unvalidated | locate bathymetry data |

## Text

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|
| X57 | LICENSE; CITATION.cff | missing before sharing | text | S | repo shareable and citable | decide: licence; then fix |
