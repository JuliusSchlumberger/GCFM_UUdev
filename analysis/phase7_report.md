# Phase 7: observed surge-river peak timing: Ebro (2433835) and Chao Phraya (3279946)

Script: `phase7_lag_cooccurrence.py`, writing `<results_dir>/figures/analysis/phase7_lag_cooccurrence_{summary,pairs}.csv` and `_hist.png`. Diagnostic only: no `lag_mode` key, and rule 14 keeps the configured lag.

## Method

- **Data:** GTSM hourly surge residual at the station nearest a mouth, and GloFAS daily discharge at the crossing with the largest mean flow, over their overlap 1979-01-02 to 2024-12-31.
- **Peaks:** above the median annual maximum, declustered over 3 days (surge) and 14 days (discharge).
- **Pairing:** each peak with the nearest peak of the other series within ±7 days. Lag = t_Q − t_surge at the boundary, in whole days because the discharge is daily.
- **Materiality:** the coastal lag (lag − T_river) is compared with τ95, the half-width of the inflow hydrograph above 95 % of its peak in `river_100_03SLR`.

## Results

| | Ebro | Chao Phraya |
|---|---|---|
| GTSM station | 3334 | 18517 |
| surge / river peaks | 35 / 44 | 25 / 39 |
| surge peaks with a river peak within ±7 days | 7 (20 %) | 0 |
| lag median (p25–p75), t_Q − t_surge | +89 h (+76 to +105) | – |
| pairs within 1 day | 0 | – |
| T_river (inflow → main mouth), τ95 | 18 h, 14 h | 26 h, 35 h |
| configured lag | −21 h | −41 h |

Peak months:

| | surge peaks | river peaks |
|---|---|---|
| Ebro | Oct–May, mostly Feb–Mar | all year, mostly Apr, Nov–May |
| Chao Phraya | Nov–Mar (NE monsoon) | Aug–Oct (SW monsoon) |

## Findings

- **Chao Phraya:** surge and river peaks are seasonally disjoint, and in 46 years no surge peak above the median annual maximum has a river peak within a week. The compound scenarios combine a NE-monsoon surge with a SW-monsoon flood, which the reanalysis does not show together. They are a design worst case, not a representative event.
- **Ebro:**
  - **Co-occurrence:** one surge peak in five is followed by a river peak, 3–4 days later (median +89 h at the boundary, +71 h at the coast after T_river). Coastal storms and the rain that raises the Ebro are the same systems.
  - **Materiality:** that lag exceeds τ95 (14 h). With the observed timing, the surge has passed before the river peak arrives.
- **Configured lags:** both (−21, −41 h) put the surge peak on the river's discharge plateau at the main mouth (`lag_plateau.py`). That is the aligned worst case, and it stays as the design setting.
  - Each scenario's methods text should say it is aligned, not observed.
  - An `observed_median` mode would only be defined for Ebro, and would remove the overlap.
