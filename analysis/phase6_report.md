# Phase 6: calibration run length and calibration sea: Ebro (2433835) and Chao Phraya (3279946)

Scenarios `coast_100_03SLR` and `river_100_03SLR`. "Walled" is the Phase 1
baseline (run length ramp + T_walled + margin), saved in
`<scratch>/phase1_baseline_snapshot`; "settle" is `results_dir` after this phase.
Figures and tables are in `<results_dir>/figures/analysis/`.

## 6.1 Run length from the diffusive settling time

Script: `phase6_settle_time.py`. Long runs: round 1 with
`run_length: {method: settle, settle_factor: 3}` and `calibration_max_days: 16`
(`<scratch>/p61_long`, `snakemake preprocess --forcerun modelled_depth_estimation`).

**Rule.** The walled travel time T_walled gives when the wave front reaches the
coast, not when the levels behind it stop rising. The settling time of a
diffusive wave over the walled profile is t_settle = L²/D, with D = Q / (2 W S)
(Hayami), the median over the mainstem nodes, and S the walled surface slope
floored at 1e-5 (`src.river_profile.settle_time_hr`). The run length is now
ramp + max(T_walled, settle_factor × t_settle) + margin.

**Measured settling** (river-bed profile nodes, 12 h mean level within 5 cm of
the end level of the long run):

| basin | T_walled | L²/D | nodes steady p50 / p90 | (p90 − ramp) / L²/D |
|---|---|---|---|---|
| Ebro | 6.2 h | 26.7 h | 23 / 24 h | 0.67 |
| Chao Phraya | 40.8 h | 114.6 h | 177 / 203 h | 1.72 |

The two basins differ by a factor 2.5, so L²/D scales the settling time
only roughly. `settle_factor: 1.75` is set by the slower basin. It is a
conservative bound, not a prediction.

![settling](<results_dir>/figures/analysis/phase6_settle_time_settle.png)

**Effect on the calibration** (level deficit = end level − level at the run end):

| basin | run length walled → settle | deficit at run end, median / p90: walled | settle |
|---|---|---|---|
| Ebro | 25 → 65 h | +0.03 / +0.04 m | +0.00 / +0.00 m |
| Chao Phraya | 59 → 219 h | **+1.87 / +5.18 m** | +0.01 / +0.03 m |

The walled Chao Phraya calibration stopped while the river was still filling.
Its crests were set up to 5 m below the steady levels.

**Steady check.** `steady_check` compares 12 h mean levels (`window_hr: 12`).
On the steep upstream Chao Phraya reach (levels 11–27 m), SFINCS levels vary
by ±0.5–2 m from hour to hour around a settled mean. Single snapshots flagged
427 cells as rising at the end of the 362 h run. With 12 h means, the 219 h
run leaves 227 of 12,462 centreline cells (1.8 %) still rising by more than 5 cm
(max +0.41 m). Ebro leaves none.

**Crests and flooding:**

| basin | crest median / p90 walled → settle | scenario | flooded area | urban exposed | volume |
|---|---|---|---|---|---|
| Ebro | 0.42 / 9.07 → 0.42 / 9.07 m | coast | 260 → 260 km² | 2.2 → 2.2 km² | 0.35 → 0.35 km³ |
| | | river | 169 → 173 km² | 3.8 → 3.9 km² | 0.23 → 0.23 km³ |
| Chao Phraya | 6.73 / 22.6 → 9.06 / 24.4 m | coast | 2037 → 1866 km² (−8 %) | 685 → 630 km² | 4.23 → 4.11 km³ |
| | | river | 2366 → 1553 km² (−34 %) | 107 → 73 km² | 5.03 → 4.45 km³ |

![crests and depth](<results_dir>/figures/analysis/phase6_settle_time_crest_depth.png)

**Other diagnostics after 6.1:**

| check | Ebro | Chao Phraya |
|---|---|---|
| 1D–2D gap, median (p5 / p95) (`phase1_gvf_gap.py`) | 0.76 m (0.18 / 1.13), unchanged | **0.56 → 3.43 m** (−0.98 / 7.12) |
| spin-up end − 1D mean level (`phase1_spinup_convergence.py`) | +0.36 m, unchanged | +1.16 m, unchanged |
| lag plateau at the main mouth (`lag_plateau.py`) | 218–257 h, lag −21.0 h | **open-ended → 315–391 h, lag −40.7 h** |

- **Chao Phraya 1D–2D gap.** The steady 2D channel stands 3.4 m (median) above
  the 1D GVF profile on the same bed. The short run hid this. It is the
  inconsistency Phase 2 addresses.
- **Lag plateau.** The higher crests close the Chao Phraya plateau, and its
  midpoint lag matches the configured −41 h. Both configured lags stay.

**Gate.** Passed. Ebro's results are unchanged: identical crests, coast flooding
identical, river flooding +2 % in the upstream valley. Chao Phraya converges:
median level deficit 0.01 m at the run end, against 1.87 m before. Config:
`run_length.method: settle`, `settle_factor: 1.75`, `steady_check.window_hr: 12`.

**Cost.** Rule 11 on 14 threads (`.snakemake/log/2026-11-01T170418.784022.snakemake.log`):
Ebro 2.0 min, Chao Phraya 98.5 min. In the Phase 1 report these were 1.3 and 33.4 min.
