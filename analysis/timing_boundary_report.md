# River peak travel and surge reach on the production model: Ebro (2433835) and Chao Phraya (3279946)

Ex-post checks on the finished production runs, advection 0 and `alpha` 0.25, scenarios
`coast_100_03SLR` and `river_100_03SLR`, built 2026-11-03. No new SFINCS runs.
Tables and figures are in `<results_dir>/figures/analysis/` and
`<results_dir>/{basin}/figures/analysis/`.

## Calibration settling with advection 0

Rule 11's `steady_check.csv`, round 1, settle run length (`settle_factor` 1.75):

| basin | run length | settled by, p95 / max of centreline cells | cells still rising at the run end |
|---|---|---|---|
| Ebro | 65 h | 21 / 21 h | 0 of 835 |
| Chao Phraya | 219 h | 170 / 171 h | 0 of 12,462 (max +0.006 m) |

The factor was measured with advection 1 (Phase 6.1: Chao Phraya p90 203 h).
With advection 0 the calibration settles sooner, and the 219 h run leaves 48 h to spare.

## River peak: inflow to mouth

Scripts: `peak_lag.py`, `lag_plateau.py` (SFINCS river-only run) and `estimate_peak_travel.py`
(1D celerities on the same event, no SFINCS). The walled time is rule 11's parametric
estimate (`walled_travel_time`, from `steady_check.csv`).

| | Ebro | Chao Phraya |
|---|---|---|
| inflow peak / peak over bankfull | 3739 m³/s / 3.9 | 9206 m³/s / 2.2 |
| SFINCS: inflow peak to mouth discharge peak (main mouth) | **16.3 h** | **24.5 h** |
| SFINCS: mouth discharge peak | 3340 m³/s | 1237 m³/s |
| SFINCS: mouth plateau (within 2 % of the maximum) | 217–251 h, closed | 307–426 h, **open-ended** |
| SFINCS: second mouth | — | peaks at the run's filter edge (+114 h) |
| rule 11 walled estimate | 6.2 h | 40.8 h |
| 1D celerity estimate, fast / regime / kinematic | 1.5 / 1.9 / 5.6 h | 5.2 / 7.0 / 32.3 h |
| lag at the plateau midpoint / configured | −17.7 / −21 h | −54.4 / −41 h |
| tau95 (hours the inflow stays within 5 % of its peak) | 10 h | 36 h |

- **Ebro.** The peak reaches the mouth 2.6 × later than the walled estimate. The peak is 3.9 × bankfull, and floodplain storage slows the wave (the walled estimate keeps all water in the channel). The difference between the plateau lag and the configured lag (3.3 h) is within tau95.
- **Chao Phraya.** The peak does not travel as a wave. Only 1237 of 9206 m³/s leaves through the main mouth at its peak; the rest goes into floodplain storage. The mouth discharge is still within 2 % of its maximum when the run ends, so the plateau midpoint (−54 h) is a lower bound. The difference from the configured −41 h (13 h) is within tau95 (36 h).
- **Consequence for production.** The walled estimate only sets the calibration run length. That length is now set by the settling time (65 and 219 h), far longer than either travel time, so the estimate's error has no effect. Both configured lags stay inside tau95 of the measured ones and need no change.

![peak lag](<results_dir>/figures/analysis/lag_plateau_coast_100_03SLR_river_100_03SLR.png)

## Surge reach up the river

Script: `surge_reach.py` (new). For each forcing point's mainstem (as in
`check_upstream_boundary.py`), it compares zsmax of the coastal run (river held at
mean discharge) with the spin-up's end level, which uses the same discharge and a
calm sea and is the event's restart state. The difference includes both tide and
surge; the storm-tide hydrograph spans the whole run, so no stretch of it is tide
only. x_inf, the threshold and the verdict follow `testing.upstream_boundary_check`:
the threshold is max(0.05 m, 0.05 × the rise at the mouth), and the margin 5 km.

| basin | forcing reach | forcing point | rise at the mouth | x_inf | margin | rise at the forcing point | verdict | surge peak arrives at x_inf |
|---|---|---|---|---|---|---|---|---|
| Ebro | 21604100071 | 65.5 km | 0.47 m | 53.3 km | 12.2 km | 0.00 m | pass | +2 h |
| Chao Phraya | 44306200155 | 264.5 km | 2.29 m | 198.3 km | 66.2 km | 0.00 m | pass | +9 h |
| Chao Phraya | 44306300031 | 293.5 km | 2.29 m | 275.4 km | 18.1 km | 0.00 m | pass | +14 h |

![Chao Phraya](<results_dir>/3279946/figures/analysis/90_surge_reach_coast_100_03SLR_profiles.png)

- **Both deltas pass on the production model.** The coastal event leaves the river level at every forcing point unchanged.
- **The reach ends abruptly.** On every path the rise holds at 0.5–2 m almost up to x_inf, then drops to zero within a few km. That is where the river surface at mean discharge rises above the storm-tide level.
- **Dynamic and steady reach are close.** The steady SFINCS benchmark (advection 1, `figs/upstream_boundary_benchmark/`) put the Chao Phraya influence ends at 200.8 and 278.9 km, against 198.3 and 275.4 km here. The 100-year storm tide lasts long enough to come close to the held-sea reach. Its timing therefore gives no reason to expect the steady check to be much too strict.
- **Smallest margin:** 18 km, on the Chao Phraya western arm.

## Open

- Steady boundary check on the current bed: `check_upstream_boundary.py` (GVF) and the SFINCS benchmark (`benchmark_upstream_boundary.py`, TODO B31). This includes the stress case (RP100 + 1 m SLR), which this event does not cover.
- The Chao Phraya river run ends before the mouth discharge plateau does (TODO B32).
