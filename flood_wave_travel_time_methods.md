# Parametric river-to-coast flood-wave travel time for GCFM_UU

Literature review and a test plan for rule-based, per-delta 1D estimates of (a) the time the river peak needs from the upstream SFINCS boundary to the mouth, `T_river`, and (b) the time the surge peak needs from the SFINCS sea boundary to the mouth, `T_surge`, both without running SFINCS. They serve two decisions:

1. **Lag.** The forcing puts the river peak at the forcing point and the surge peak at the sea boundary at the same time. For the peaks to co-occur at the mouth, the river series has to start `lag = T_river − T_surge` earlier, if that lag is material.
2. **Run length.** The event run must contain the time at which the river crest has passed the mouth, shifted or not.

Status: planning document. Diagnostics on finished event runs and the main-path profile exist (§0.1); the parametric estimates are not built.

---

## 0. Scope and the quantities to estimate

The same per-reach celerity model yields several times. They are not interchangeable, and they need opposite bounds.

| Symbol | Definition | Used for | Bound wanted |
|---|---|---|---|
| `T_lag` | Peak-to-peak time, inflow discharge peak → river-driven discharge peak at the mouth (`T_river`) | Shifting the hydrograph so peaks coincide at the coast | Best estimate (P50) |
| `T_surge` | Surge peak at the SFINCS sea boundary → mouth (shallow-water wave over the bathymetry) | Subtracted from `T_lag`: the lag to apply is `T_lag − T_surge` | Best estimate |
| `T_arrive` | Time until the rising limb is felt at the mouth (first significant rise, e.g. 10 % of the eventual rise) | Start of the window: river must already be rising at the coast when the surge arrives | Fast estimate (P10) |
| `T_transit` | Time until the routed hydrograph at the mouth has fallen back below a threshold (95 % of its peak for "peak has passed"; bankfull/protection level for "flood is over") | End of the window: the max water level in the channel has occurred | Slow estimate (P90) |
| `T_fill` | Time to fill floodplain storage between bankfull and flood stage from the overbank discharge | End of the window on the floodplain: the max depth off-channel has occurred | Slow estimate |
| `T_adjust` | Time for a steady discharge to reach a steady profile (spin-up, dike-crest calibration runs) | Spin-up and calibration run length | Slow estimate, then checked in-run |

**Lag.** `T_lag(basin, RP) − T_surge(basin)` lets the river hydrograph be shifted earlier so its peak coincides with the surge peak at the mouth rather than at the two boundaries. The hook exists: `sfincs.boundary_setup.compound.lag_hr` (one value for every basin, default 0) shifts the river series in `14_build_sfincs.py`.

**Modelling period.** The event run starts from the spin-up restart and lasts `2 × window_hr`, symmetric around the surge peak (`boundary_forcings.surge.hydrograph.window_hr`, default 60 h; per-basin `window_overrides_hr`, 168 h for the Ebro). The forcing files also carry the `lead_days` lead-in, which the spin-up covers (`tstart = tref + spinup_days`, so `spinup_days` must equal `lead_days`). The rule in §2.2 replaces the overrides with an asymmetric window derived from `T_arrive`, `T_transit`, `T_fill`, the surge duration and the tidal spin-up.

**Constraints.**
- No new SFINCS runs for the estimate itself; finished event runs serve as validation (§0.1). Inputs:
  - rule 11's river-bed node table `river_bed/{basin}_river_bed_nodes.csv` (`reach_id`, `dist_from_mouth_m`, `dist_to_sea_m`, `width_m`, `slope`, `bed_m`, `level_m`, `q_design_m3s`; nodes 500 m apart), read through `analysis/_upstream_profile.py`;
  - the river network `{basin}_river_network_depth_estimated.gpkg` (`is_seed`, `rch_id_dn`, `river_depth_m`, `river_bed`, `bankfull_discharge_m3s`);
  - `river_forcing.nc` (mean and return-period discharges, daily GloFAS series);
  - the subgrid DEM, the GTSM average tide and surge shapes (`surge_forcing.nc`), and `channel_manning_n` (0.03).
- SFINCS ground truth on the Ebro (2433835): 50-yr flood peak lag 1.5 / 2.0 / 2.2 / 2.3 h at 4 / 12 / 19 / 27 km from the inflow, i.e. ~8 m/s from 4 to 27 km; bed 3.6–8 m below sea level along the whole domain (source run to be cited). `check_event_timing.py` gives 1.25 h gravity-wave travel over the 47.4 km path in the current run. Keep this as the first regression test for every method.
- Output must be a value plus a *materiality* flag: for flat hydrographs a lag of a few hours changes nothing, and applying it only costs lead-in time.

**What the literature does not provide.** No published global compound-flood workflow uses a parametric per-delta lag. Ward et al. (2018) and Couasnon et al. (2020) search statistically for the lag that maximises dependence (up to ±5 days, daily data); Eilander et al. (2020, 2023) let the routing model (CaMa-Flood) deliver discharge at the coastal boundary, so the lag is absorbed by the routing and never stated. Allen et al. (2018) is the only global, parametric travel-time product, and it is a kinematic wave on bankfull geometry that explicitly ignores backwater and tides. Dykstra & Dzwonkowski (2020) is the only observational study of a fluvial flood wave crossing a full backwater–estuary transition. The method we need sits between those two and has to be assembled and validated here.

### 0.1 What exists and what the finished runs show

Built:
- `analysis/check_event_timing.py`: per finished event run, the time of max water level of every wet cell relative to the surge peak (hourly `zsmax` with `timemax`, `dtmaxout: 3600`), τ95/τ90 of the summed inflow, the M1 gravity-wave travel time from the main inflow to the open boundary along wet cells at flood stage, the materiality flag (M1 vs τ95) and late-peak shares. Writes `figs/event_timing/event_timing.csv` and one map per basin.
- `analysis/_upstream_profile.py`: main path from a forcing reach to the sea (largest discharge at bifurcations), the node profile along it, and `forward_levels`, the steady GVF level for any discharge and sea level (`src.gvf_solver.integrate_reach`). This covers §3 steps 1–3 and gives the surface slope M3 needs.
- `src.river_network` routes discharge through the network with a width split at bifurcations.
- `analysis/estimate_peak_travel.py`: the 1D estimate, from build inputs only (node table, river network, skeleton `gis/`, the scenario's forcing files).
  - River leg: the steady GVF level at the scenario's peak main inflow, with the sea at the storm-tide peak level, along the main inflow's path. Three celerities per node: fast `u + √(gh)` (M1), slow `5/3·u` (kinematic, M2 on the surface slope), and regime (fast where the bed is below the calm sea level, slow elsewhere; a reduced M4).
  - Sea leg: the fastest wet path from the skeleton's water-level boundary cells to the mouth at `√(gh)`.
  - Writes the lag, its materiality (lag > τ95) and the run length needed with and without the shift to `figs/peak_travel/`. Tests in `tests/unit/test_peak_travel.py`.

1D estimate, scenario `default` (only these basins have a current rule 11 node table):

| basin | path km | drowned | Q_peak/Q_bf | T_surge h | τ95 h | T_river fast / regime / slow h | lag material (slow) | window ok |
|---|---|---|---|---|---|---|---|---|
| 2209166 | 41.0 | 0.27 | 3.6 | 0.24 | 4.0 | 0.7 / 1.1 / 1.3 | no | yes |
| 2433835 | 64.5 | 0.79 | 3.3 | 0.15 | 19.5 | 1.6 / 2.0 / 6.3 | no | yes |
| 3279946 | 288.4 | 0.92 | 2.0 | 0.17 | 15.5 | 5.1 / 7.0 / 35.6 | yes (35.4 h > 15.5 h) | no, if shifted: needs 89 h before the surge peak, has 60 h |

Readings:
- **Surge leg.** The sea boundary lies 4.5–8.8 km from the mouth, so `T_surge` is 0.15–0.24 h and negligible against `T_river`.
- **Ebro.** The regime value (2.0 h) matches the SFINCS reference (≈ 2 h).
- **Deciding basin.** 3279946 (288 km, 92 % drowned) is the only one where the choice of celerity flips the decision: dynamic/regime says immaterial, kinematic says a 35 h lag. With `λ = c·T_event` of thousands of km against a 288 km path the quasi-static (regime) reading is more likely (§1.3), but the Mobile delta (Dykstra & Dzwonkowski 2020) shows large events can be diffusive and slow. A river-only SFINCS run of 3279946 settles it (TODO B32).
- **Channel-only bias.** All three peaks are 2–3.6× bankfull and the estimate keeps the whole peak in the channel, so the slow celerity is too fast where overbank storage matters (2209166: in-channel velocities of 4–9 m/s on a steep bed). The overbank correction (M5) is the next refinement; for 2209166 it could bring the slow lag (1.05 h) close to τ95 (4 h).

Finished event runs, scenario `default` (lag 0):

| basin | path km | M1 travel h | τ95 h | p95 time of max after surge h | share peaking in last 6 h | late rise p99 m | run h |
|---|---|---|---|---|---|---|---|
| 1248635 | 201.5 | 5.3 | 27.8 | +60 (run end) | 0.15 | 1.87 | 120 |
| 1475442 | 86.0 | 3.4 | 43.0 | +60 (run end) | 0.18 | 0.56 | 120 |
| 2209166 | 61.6 | 1.7 | 4.0 | +1 | 0.00 | 0.00 | 120 |
| 2433835 | 47.4 | 1.3 | 11.5 | +1 | 0.00 | 0.00 | 336 |
| 2444235 | 29.1 | 1.0 | 1.8 | +60 (run end) | 0.18 | 0.68 | 120 |
| 620947 | 32.6 | 1.1 | 24.0 | +11 | 0.00 | 0.00 | 120 |

Readings:
- **Lag.** M1 is below τ95 in all six, but M1 is the fastest bound. On the long paths a diffusive celerity could make the lag material (1248635: 201.5 km at 1 m/s is 56 h against τ95 = 27.8 h). The river-path observation points of the same runs give the SFINCS peak lag directly and settle this without new runs (E1).
- **Run length.** In three of six the time of max sits at the run end for 15–18 % of rising cells. In the maps (1248635, 1475442) these are enclosed floodplain compartments away from the channel; the channel peaks within about ±10 h of the surge peak. The compartment check (`event_timing_compartments.csv`) shows they are filling, not river transit:
  - 1248635: 168 compartments, 704 km², 96–100 % of their volume gained after the surge peak, still taking 1335 m³/s at the run end; the largest (220 km²) rises 1.4 cm/h. The max level there is not reached in the run.
  - 1475442: 174 km², 76–81 % gained after the surge peak, rising ≤ 0.5 cm/h at the end (near their final level).
  - 2444235: one 40 km² compartment not bordering the channel (fed through other floodplain), rising 0.1 cm/h.

  So `T_fill` controls `after` in these basins, and in 1248635 it exceeds the 60 h after the surge peak the run has. Caveat: these three runs predate the current preprocessing (TODO B30).
- **Ebro.** At 168 h the p99 time of max is +2 h, so the override is far longer than the run needs.

Consequence for the order of work:
1. Validate the river leg against a river-only SFINCS run on 2433835 and 3279946 (TODO B32).
2. Add the overbank correction (M5) to the slow celerity.
3. Rebuild the remaining basins so they get node tables (TODO B30), then screen them all (E2).

Floodplain filling (`T_fill`) stays a check on finished runs (`check_event_timing.py`); the 1D channel estimate cannot see it.

---

## 1. Literature review

### 1.1 Flood-wave celerity in rivers (classical theory)

- **Kleitz–Seddon law.** The flood-wave celerity is `c = dQ/dA = (1/W) dQ/dh` (Kleitz 1877; Seddon 1900). With Manning in a wide channel `c = (5/3) v`; with Chézy `c = (3/2) v`. In natural channels the ratio `c/v` is typically 1.3–1.7 in-bank; the USGS still uses this to time-shift discharge measurements made away from a gauge (USGS Office of Surface Water memo 92.09).
- **Wave hierarchy (Lighthill & Whitham 1955; Ponce & Simons 1977).** Linear stability analysis of the Saint-Venant equations gives, as a function of a dimensionless wavenumber `σ* = (2π/λ)·(h/S0)`, a spectrum from kinematic (gravity–friction balance, celerity ≈ Seddon, no attenuation) through diffusive (small attenuation, same celerity) and "mixed" (strong attenuation, rapidly varying celerity) to dynamic/gravity waves (celerity → `v ± √(gh)`, little attenuation). The important practical facts: (i) a flood wave is a kinematic wave with a little diffusion (Ponce 2023 review); (ii) in the middle of the spectrum celerity is up to an order of magnitude faster than kinematic and attenuation peaks; (iii) mild slopes push a wave toward the dynamic end.
- **Applicability criteria (Ponce, Li & Simons 1978; Ponce 1989).** With `T` the wave period (≈ hydrograph duration), `S0` bed slope, `u0`, `d0` the reference velocity and depth:
  - kinematic: `N = T·S0·u0/d0 ≥ 85` (practical; ≥171 for 95 % accuracy after one period),
  - diffusive: `M = T·S0·√(g/d0) ≥ 15` (practical; ≥30 in the 1978 paper).
  Both require a non-zero bed slope; they are undefined for a drowned channel and that is itself the diagnosis.
- **Diffusion effect on peak timing.** A diffusive wave with a flat crest arrives *earlier* than `L/c` because the crest flattens; the peak-to-peak lag from routing is shorter than the celerity-based lag. Conversely the downstream peak is smaller (attenuation). For hydrographs that are 74–100 % of peak for a day either side of the crest (our GloFAS floods), the peak-to-peak lag is poorly defined at the scale of hours.
- **Wave speed vs discharge, in-bank to overbank.** `c(Q)` is not monotonic: it rises through the in-bank range, drops sharply at bankfull as floodplain storage increases `dA/dQ`, and recovers once the floodplain is fully engaged (Wong & Laurenson 1983; Sriwongsitanon et al. 1998; Tang, Knight & Samuels 2001; HR Wallingford SR568). Any method using a single in-channel `c` overestimates speed for `Q_rp > Q_bankfull`.
- **Backwater on routing.** Tsai (2003, 2005) and Chung, Aldama & Smith (1993) extend the applicability criteria to a downstream boundary condition: a fixed downstream stage (sea level) reflects part of the wave and changes celerity and attenuation near the boundary. Tsai (2005) gives analytical solutions in terms of the Froude number and `k* = Lb/λ` (see §1.3).

### 1.2 Practical travel-time estimation without a hydrodynamic model

- **Jobson (USGS WRI 00-4187, 2000; JHE 2001).** Travel time of water (not of the wave) from hydraulic characteristics: active flow area from Manning with `n = 0.035` plus a constant inactive area; ~10 % error for 60 reaches with `S > 0.0002`. Useful for the velocity `v` that enters `c = k·v`, but it is solute travel time, and it explicitly excludes slopes below 2·10⁻⁴, which is where our deltas live.
- **Allen et al. (2018, GRL).** Global flow-wave travel time: kinematic wave on the HydroSHEDS network with bankfull width/depth from hydraulic geometry (Andreadis et al. 2013), 15-arcsec slopes with a minimum-slope floor, Monte Carlo over Manning's `n`, slope and geometry. Median travel time to basin outlet 6 days globally. Validated against lagged cross-correlation of USGS gauge pairs; the model is systematically *faster* than gauges because it computes the maximum (bankfull, in-bank) speed. Limitations stated: no backwater, tides, floodplain storage, reservoirs. The Zenodo dataset (10.5281/zenodo.1015799) includes modelled celerity per reach (`CELER_MPS`) and gauge-based celerity (`OBS_CEL_MPS`) — a free benchmark for our method where our deltas overlap North America.
- **Operational crest-travel tables (NWS/USACE).** For the lower Mississippi, NWS crest forecasts in March 2025 placed the crest at Vicksburg on 17 March and at Baton Rouge on 20 March, i.e. ~333 km in 3 days ≈ 1.3 m/s; Memphis → Baton Rouge ≈ 8 days ≈ 1.2 m/s. These are slower than `1.5·v` at flood stage and are strongly affected by spillway operations (Bonnet Carré, Morganza), which in 2011 made Baton Rouge crest *before* Vicksburg. Use them as order-of-magnitude checks, not as validation targets.

### 1.3 Backwater, tidal rivers, and fluvial waves in deltas

- **Backwater length.** `Lb = h/S` (Paola & Mohrig 1996) is the standard scale for the reach where the river profile is controlled by the receiving basin; Mississippi ≈ 500 km (Lamb et al. 2012; Nittrouer et al. 2012). Input conventions matter: bankfull thalweg depth from a cross-section rather than depth from hydraulic geometry (Wolf et al. 2024 review). The pipeline does not compute `Lb` (the domain comes from the delta polygons); compute it from rule 11's bed profile, and compare it with the sea-influence length of `analysis/check_upstream_boundary.py`.
- **Tidal river dynamics (Hoitink & Jay 2016, Rev. Geophys.).** River discharge damps the tide and raises the subtidal water level; the tidal reach extends hundreds of km inland in large rivers. Kästner et al. (2019, JFM) show that on a sloping bed the point where the bed reaches sea level effectively limits tidal intrusion — a clean criterion for the drowned/tidal reach. Cai, Savenije & Toffolon (2012, 2014) give an analytical framework (four implicit equations for damping, velocity amplitude, celerity and phase lag) that includes river discharge; it is written for the *tidal* wave, not for the fluvial flood wave, but gives the tidal celerity and the river-to-tidal velocity ratio needed to classify a reach as bidirectional.
- **Dykstra & Dzwonkowski (2020, WRR) — the key paper.** Fluvial flood waves tracked over 238 km through the Tombigbee–Alabama delta and Mobile Bay (Lb = 151 km) with water level and velocity at 22 stations. Findings that bear directly on our problem:
  - The wave type is set by `k* ≈ Lb/λ`, with `λ` the wavelength (`c·T_event`). From Tsai (2005) at `Fr ≈ 0.1`: diffusive for `k* < 1/3`, dynamic for `k* ≈ 1/3–2`. Observed `k*` was 0.43 / 0.29 / 0.14 for 600 / 1000 / 2000 m³/s events, so small events were dynamic and large ones diffusive; the diffusive→dynamic transition moved seaward with event size and coincided with the transition from unidirectional to bidirectional (tidal) flow.
  - In-channel events (< 2000 m³/s) sped up with discharge, reaching ~0.9 m/s across the delta; `c = 1.5·u` was 6–27 % too fast. In the bay small events moved at ~4.4 m/s, near the tidal-wave speed.
  - Overbank events (5000–8000 m³/s) moved at only 0.26–0.30 m/s across the lower delta — three times *slower* than in-channel events — because floodplain storage increased `dA/dQ`; `c = 1.5·u` overestimated celerity four-fold. Lag time increased with discharge between 2000 and 5000 m³/s and was constant above.
  - The velocity (horizontal) peak led the water-level (vertical) peak by more than a day in the backwater, so "peak at the coast" must specify which variable.
  - The authors state that the usual estuarine practice of a constant lag is a first-order approximation only, and recommend diffusive-wave routing for subtidal discharge in tidal rivers.
- **Implication for drowned, short deltas (Ebro).** When `λ ≫ L_domain` (Ebro: `c ≈ 8 m/s`, `T ≈ 2–3 days` → `λ ≈ 1500–2000 km` against a 40 km domain) the entire channel responds quasi-statically: the lag is the gravity-wave transit `L/√(gh)`, hours at most, and immaterial against a daily hydrograph. This is what the Ebro SFINCS run showed.

### 1.4 Compound-flood practice

- Ward et al. (2018): dependence between GESLA skew surge and GRDC discharge rises from 22–36 % to 54–56 % of stations when a lag of up to 5 days is allowed — the lag matters statistically, and it is an unknown they search over, not a quantity they estimate.
- Couasnon et al. (2020): same with modelled daily GTSR surge and daily discharge at river mouths; GloFAS-style routing supplies the mouth discharge.
- Eilander et al. (2020): CaMa-Flood coupled to GTSM at daily and hourly resolution; compound-dominant deltas are those with large surges and fast-responding or flat basins. Eilander et al. (2023): SFINCS set up from global data with CaMa discharge at the upstream boundary and GTSM at the coast; the time lag between drivers is discussed as an event property, and the routing model carries the river wave to the boundary.
- Conclusion: in the global literature the river-to-coast lag is either delegated to a routing model or searched statistically. A defensible parametric method therefore has to be validated against (i) routing-model internal lags (GloFAS inflow cell → coastal cell), (ii) gauge/crest observations, and (iii) hydrodynamic runs where they exist.

### 1.5 Synthesis

| Regime | Diagnostic | Celerity | Reference |
|---|---|---|---|
| Steep, in-bank, slow hydrograph | `N ≥ 85` or `M ≥ 15`, `Q_rp < Q_bf` | `c = (5/3)(1/n) h^(2/3) S^(1/2)` with bed slope | Ponce 1989; Allen 2018 |
| Backwater (M1 profile), unidirectional | `dist_from_mouth < Lb`, `k* < 1/3`, `u_river > u_tide` | diffusive, `c ≈ 1.5·u` with *surface* slope | Tsai 2005; Dykstra 2020 |
| Overbank in backwater | `Q_rp > Q_bf` | 3–4× slower than in-channel `1.5·u` | Dykstra 2020; Tang 2001 |
| Tidal/bidirectional or drowned | bed below MSL, `u_tide > u_river`, `k* ≳ 1/3` | dynamic, `c ≈ √(gh) ± u`; quasi-static if `λ ≫ L` | Dykstra 2020; Kästner 2019 |

The regime boundaries move with event size, so the classification must be done per return period, not per delta.

---

## 2. Candidate methods

All methods operate on the main river path from the main forcing reach to the mouth, node by node (500 m), and sum `L_i / c_i`. Define per node: `L_i`, `W_i`, `z_bed,i`, `h_i(Q)`, `S_bed,i`, `S_surf,i(Q)`, `n_i`, `Q_i(Q_rp)` (after bifurcation partitioning), `Q_bf,i`.

| ID | Method | Celerity per reach | Inputs | Expected bias | Where it should work |
|---|---|---|---|---|---|
| M0 | Zero lag | — | — | — | Any delta with `λ ≫ L_domain` |
| M1 | Gravity wave | `√(g·h_i)` | `h_i` at flood stage | Lower bound on time; too fast in diffusive reaches | Drowned/tidal reaches; short deltas |
| M2 | Kinematic, bed slope (Allen-type) | `(5/3)·(1/n)·h^(2/3)·S_bed^(1/2)`, `h` from Manning at `Q_rp` | `S_bed` from bed profile, `n`, `W` | Undefined/absurd where `S_bed ≈ 0`; max speed | Upstream non-backwater reaches only |
| M3 | Diffusive, surface slope | as M2 but with `S_surf,i(Q_rp)` from a steady GVF profile | GVF solution (`_upstream_profile.forward_levels` at `Q_rp`) | Still in-channel; no overbank correction | M1-profile backwater reaches |
| M4 | Regime blend | M3 if `k* < 1/3` and unidirectional; M1 if bidirectional/drowned; transition by `k*` and `u_tide/u_river` | `Lb`, `T_event`, tidal velocity amplitude | Depends on regime diagnostics | Whole path |
| M5 | M4 + overbank correction | `c_i × (A_ch/A_active)` or compound-channel `c(Q)` when `Q_i > Q_bf,i` | floodplain width from DEM, `Q_bf` | Correction factor uncertain (3–4× in one study) | Large-RP events on wide floodplains |
| M6 | Routing-based lag | route the actual scaled hydrograph with a diffusive-wave (Hayami) or Muskingum–Cunge scheme using the M4/M5 celerity and diffusivity `D = Q/(2·W·S_surf)`; read peak-to-peak lag | same as M5 + hydrograph | Resolves the "flat crest arrives early" effect and gives attenuation | Long deltas (Mississippi-class) |
| M7 | Data-driven (GloFAS) | lagged cross-correlation and event-wise peak-to-peak lag between the GloFAS inflow cell and the most downstream GloFAS cell (Allen's validation technique) | daily GloFAS series | Daily resolution; LISFLOOD kinematic routing with its own biases | Independent check for lags ≥ 1 day |

Notes.
- M2 is included as the "what Allen 2018 would give" reference, not as a candidate for backwater reaches. Its failure on the Ebro (35 h at mean flow against a 12 h fill that came from the sea) is already documented.
- M3 is the physically defensible in-channel estimate for backwater: the flood wave moves on the water-surface slope, not the bed slope. `forward_levels` re-solves the steady level along the main path for any discharge, so `S_surf(Q)` per node is available at no extra cost.
- M6 is still parametric (1D, seconds to run) and is the one that answers "when does the *peak* arrive" rather than "how fast does the wave move". It is the method Dykstra & Dzwonkowski recommend for subtidal discharge in tidal rivers. It is also the only method that gives `T_arrive` and `T_transit` directly, because it produces the whole routed hydrograph at the mouth rather than a single celerity; M1–M5 only give `T_lag` and need the hydrograph shape added back by hand (`T_transit ≈ T_lag + τ_recession`).
- The Ponce criteria `N`, `M` and `k*` are diagnostics, computed for every method and stored with the result.

### 2.1 Materiality rule (applies to every method)

From the scaled GloFAS hydrograph `Q(t)` for the return period, compute `τ95`: the time before the peak at which `Q` is still ≥ 95 % of `Q_peak`. If `T_lag ≤ τ95`, the lag is immaterial (the river is within 5 % of its peak at the coast when the surge peaks); set `lag_hr = 0` and record `T_lag` and `τ95` in the provenance. Otherwise apply `T_lag`. A stricter variant uses the water-level sensitivity: `ΔQ → Δζ` through the local rating at the mouth; keep the 95 % discharge version unless tests show the rating is strongly nonlinear.

### 2.2 Modelling-period rule

The event run must contain, for every point in the domain, the time at which the maximum water level occurs. Three processes can put that time after the surge peak: the river crest arriving late (if the lag is not applied or is underestimated), the overbank wave crossing the floodplain (0.26–0.30 m/s in the Mobile delta, i.e. ~4 days per 100 km), and the floodplain filling. Before the surge peak the run must contain the tidal spin-up and enough of the rising limb that the river is already up at the coast.

Per delta and return period, with `t = 0` at the surge peak (which sits on tidal high water):

```
before = max( 2 * T_tide,                      # tidal spin-up on top of the calm spin-up restart (12.42 h semidiurnal, 24.84 h diurnal)
              T_surge_half,                    # time the GTSM surge shape is above 50 % of its peak before the peak
              T_lag + τ_rise )                 # shifted rising limb: from Q crossing Q_bf (or 50 % of peak) to the peak, plus the lag

after  = max( T_surge_half,                    # surge tail
              T_transit_P90 - T_lag,           # routed Q at the mouth back below 95 % of peak, slow estimate, measured from the surge peak
              T_fill,                          # floodplain storage / overbank inflow
              T_overbank_P90 )                 # slowest floodplain wave across the floodplain length: L_fp / c_overbank (c = 0.25–0.5 m/s)

after  = max(after, after_min)                 # 24 h floor so the recession is visible in sfincs_his
window_before_hr = ceil(before)
window_after_hr  = ceil(after)
event_hours      = window_before_hr + window_after_hr
```

Definitions:
- `τ_rise`: from the scaled GloFAS hydrograph, time between `Q` crossing `max(Q_bf, 0.5·Q_peak)` on the rising limb and the peak. Using the full hydrograph from `Q_mean` is unnecessary — the ramp from mean discharge belongs to the spin-up.
- `T_fill = V_fp(RP) / (Q_rp − Q_bf)` with `V_fp` the DEM volume between bankfull stage and the flood stage over the valley bottom within `Lb` (per reach, summed). It is a lower bound on the time to reach the max floodplain depth; if `T_fill` dominates the run length, the delta is one where the floodplain, not the channel, controls timing.
- `T_overbank`: only where `Q_rp > Q_bf` over a substantial fraction of the path; `L_fp` is the along-valley length of the overbank reaches. Use the Dykstra range as the prior until E4 refines it.
- `T_transit_P90`: from M6 with the slow Monte Carlo parameters (high `n`, overbank factor 5); from M1–M5 as `T_lag_P90 + τ_recession` where `τ_recession` is the time from the peak back to 95 % of the peak on the scaled hydrograph.
- The lead-in used to build the forcing stays `before`; the river forcing itself starts at `t = −(before + T_lag)` so the shifted rising limb is complete, which is the extension of `lead_days` by `lag_hr` noted in §5.

Why asymmetric: in a short drowned delta `before` is set by the tide (~25–50 h) and `after` by the surge tail or the 24 h floor, so the run is near the current defaults. In a Mississippi-class delta `after` can be several days because of `T_transit` and `T_fill`, while `before` need not grow at all beyond `T_lag + τ_rise`. A symmetric window pays for both sides.

Cost control: `event_hours` is reported per delta with which term set each side, so the expensive deltas are visible and the term responsible can be revisited (e.g. tighten the 95 % threshold to 90 %).

Dike-crest calibration runs (rule 11, rounds 1–2), implemented:
- **Forcing.** Rivers start at the steady mean-discharge level on the bed (as in the skeleton's `zsini`) and ramp from the mean to the protection-RP discharge over `ramp_hr` (6 h); the sea ramps from the calm level to the storm tide at the coastal protection RP. Both are held to the run end.
- **Run length.** `ramp_hr + T_walled + calibration_margin_hr` (12 h), where `T_walled` is the slowest walled kinematic travel time (`5/3·u`, all flow in the channel) from a crossing to its network end (`src.river_profile.walled_travel_time`). Runs longer than `calibration_max_days` (4) fail, and `steady_check` warns when levels still rise at the run end.
- **Benchmark (old 24 h round 1 runs).** On 2433835, SFINCS settled within 1–4 h of `ramp + T_walled`; the gravity wave was far too fast on all three basins. On 3279946, the upper 150 km had not settled after 24 h.
- **New run lengths.** 2433835: 25 h, 2209166: 20 h, 3279946: 59 h (its 257 km tributary, `T_walled` 40.8 h, sets it). `T_adjust` for a drowned path is a few gravity transits once the initial level is set to MSL; for a long backwater path it is the diffusive adjustment time `L²/D` with `D = Q/(2·W·S_surf)` — report both and take the larger.

---

## 3. Data preparation per delta

Build one table per basin with one row per node (500 m) along the main path; steps 1–3 are `_upstream_profile.mainstem_profile` plus `forward_levels`.

1. **Main path.** `mainstem_path` from the main forcing reach (largest inflow) to the network end, taking the larger routed discharge at bifurcations (`src.river_network` width split); record the other branch for later.
2. **Bed profile.** `bed_m` and `slope` from rule 11's node table, the same bed the SFINCS grid is burned with. `S_bed` from a monotone (non-increasing downstream) fit of `bed_m(dist_from_mouth_m)`, floored at `bed.slope_floor` (1e-5). Flag nodes with `bed_m < calm_sea_level_m` as drowned.
3. **Hydraulics.** `width_m` from the node table (SWORD widths, smoothed); `n = channel_manning_n`; depths:
   - `h_mean`: `level_m − bed_m` (the FathomDEM surface is the mean-flow level);
   - `h_bf`: Manning depth at bankfull on the surface slope (rule 11's round-0 bed, `design_discharge: bankfull`);
   - `h_rp`, `S_surf(Q_rp)`: `forward_levels` at `Q_rp` with the calm sea level at the mouth.
4. **Backwater and tidal scales.** `Lb = h_bf/S_bed` using the upstream (non-drowned) bed slope and bankfull depth; `L_tidal`: reaches where the tidal velocity amplitude (`u_tide ≈ a·√(g/h)·(convergence factor)`, or simply from the GTSM tidal range at the mouth and the Cai–Savenije damping) exceeds `u_river(Q_rp)`. Record `L_domain` = `dist_from_mouth_m` of the first node of the path.
5. **Hydrograph shape.** For each RP: `T_event` (duration above 50 % of peak), `τ95`, `τ90`.
6. **Bankfull and floodplain.** `Q_bf,i` from `bankfull_discharge_m3s` routed through the network (`bankfull_discharge_acc_m3s`); overbank flag `Q_i(Q_rp) > Q_bf,i`; floodplain width from the DEM at the flood stage (count of cells below stage within the valley, per reach); floodplain storage `V_fp,i(RP)` = Σ over valley cells of `max(0, stage_rp − max(z_cell, stage_bf)) × cell area`, with `stage_rp` and `stage_bf` interpolated along the GVF profile. `L_fp` = Σ `L_i` over overbank reaches.
7. **Surge and tide shape.** From the GTSM-based normalised surge shape per RP: `T_surge_half` before and after the peak (time above 50 % of peak). Dominant tidal period from the GTSM average tide in `surge_forcing.nc` (spectral peak, or the high-water spacing).
8. **GloFAS cells.** Index of the GloFAS cell at the inflow and of the most downstream cell on the same river (for M7).

---

## 4. Test plan

Order: E1 (river-only runs on 2433835 and 3279946 against `estimate_peak_travel.py`) → M5 overbank correction → E2 on every rebuilt basin → E5. E3, E4 and E6 follow once a basin needs them.

### E1 — Regression on the finished event runs (Ebro first)

Targets: Ebro 50-yr lag 1.5–2.3 h over 4–27 km (≈ 8 m/s); mean-flow settling driven by the sea, not by the river.
Expected outcome: M1 and M4 ≈ 2 h; M2/M3 give multi-hour values that are wrong for the right reason (`S ≈ 0`); M0 passes the materiality rule. Any method that does not reproduce ~2 h on the Ebro is disqualified for drowned deltas.
Lag on all six: the time of max water level at the river observation points (`sfincs_his.nc`) along the main path gives the SFINCS peak lag per km; compare it with M1 and M3.
Late peaks: split the cells peaking in the last 6 h into channel and off-channel, group the off-channel ones into connected compartments, and per compartment read whether its volume is still rising at the run end and where it gains water (overtopped weir edges, gaps). If they are filling compartments, `T_fill` is the term to estimate; compare `V_fp / Q_in` per compartment with the time the compartment needs in the run.
Modelling period: the §2.2 rule must put every observation point and, apart from the compartments still filling at the run end, every cell inside `[−before, +after]` with margin. Compare the rule's `event_hours` with the current `2 × window_hr` (Ebro 2 × 168 h) and record which term set each side. Also read off from the his file where the max occurred relative to the surge peak at the upstream points — if it is after the surge peak, the current symmetric window is being carried by the river tail, which is the case the rule is meant to size.

### E2 — Cross-delta screening

Run all methods for every basin with a network file. Deliverable: one table (`timing/screening.csv`) with per basin and RP: `L_domain`, `Lb`, `L_tidal`, fraction of path drowned, `k*`, `N`, `M`, overbank fraction, `T_lag` for M0–M7, `τ95`, materiality flag per method, and the modelling-period terms (`T_arrive`, `T_transit_P90`, `T_fill`, `T_overbank_P90`, `T_surge_half`, `T_tide`, `before`, `after`, `event_hours`, controlling term per side).
Questions to answer:
- For how many deltas do the methods disagree by more than `τ95`? Those are the only ones where the choice matters.
- Does `k*` separate the deltas into the quasi-static and diffusive groups cleanly, or is there a large ambiguous band?
- Distribution of `L_domain/Lb` — if most domains are entirely within `Lb`, M2 is irrelevant everywhere and the question reduces to M1 vs M3/M6.
- How does `event_hours` compare with the current `lead_days + 2 × window_hr` per delta: where is the default too short (risk: max water level after the run ends — check by whether the his-file max sits at the last time step in any existing run), and where is it longer than needed (cost)? Which term controls `after` most often — surge tail, `T_transit`, or `T_fill`?
- Total run-time budget: Σ `event_hours` over deltas and RPs with the rule vs with the defaults.

### E3 — GloFAS internal lag (M7) as an independent check

For deltas with `L_domain > 150 km` (daily resolution is useless below that): lagged cross-correlation of the full daily series, plus event-wise peak-to-peak lags for the 20 largest events, between the inflow cell and the downstream cell. Compare with M3–M6. Report the ratio and whether M7's uncertainty (±0.5 day) covers the parametric value.
Caveat to state in the write-up: GloFAS routing (LISFLOOD kinematic wave) has no backwater either, so agreement with M2 is not evidence of correctness in backwater; disagreement with M6 in a strongly backwatered reach is informative in the other direction.

### E4 — Observational benchmarks

- Mississippi: NWS/USACE crest dates between Vicksburg, Natchez, Red River Landing, Baton Rouge, Donaldsonville, Reserve and New Orleans for 3–5 non-spillway floods; convert to celerity; compare with M3/M5/M6 for the matching discharge. Expect observed ≈ 1.2–1.5 m/s.
- Tombigbee–Alabama: the Dykstra & Dzwonkowski celerities (0.26–0.91 m/s in the delta, 4.4 m/s in the bay) as a published test case; even without building that delta in SFINCS the reach table can be assembled from SWORD/DEM and the methods run.
- Allen et al. 2018 `OBS_CEL_MPS` (gauge-pair celerity) for any North/Central American reach that overlaps a GCFM delta path.
- Ebro at Tortosa vs any downstream gauge if GRDC has one; otherwise the SFINCS run stands.
- Where GRDC pairs exist on the lower reach of any modelled delta (Rhine Lobith–downstream, Rhône Beaucaire–Arles, Po Pontelagoscuro, Mekong Kratie–Tan Chau, etc.), compute gauge-pair lags with the Allen cross-correlation code and add them to the benchmark table.

Metrics: log-ratio `ln(T_pred/T_obs)`, fraction within a factor 2, and — most important — agreement on the materiality flag (would the method and the observation make the same apply/skip decision).

### E5 — Uncertainty

Monte Carlo per delta: `n ∈ [0.025, 0.045]`, depth ±30 %, slope floor `[1e-6, 1e-5]`, overbank factor `[2, 5]`, `T_event` ±30 %. Report P10/P50/P90 of `T_lag`. Decision rule uses P50; if P10–P90 straddles `τ95` the delta is flagged "lag uncertain, immaterial by default" and listed for a SFINCS river-only run when runs become possible again.

### E6 — Sensitivity of the compound result (deferred, protocol fixed now)

When SFINCS runs are allowed: river-only diagnostic run (calm sea, same hydrograph), measure peak-discharge timing at cross-sections along the path and the time of maximum water level in every cell (`sfincs_map.nc`, `zsmax` time or a per-step comparison); then compound runs with `lag = 0` and `lag = T_lag`; compare max-water-level fields (95th percentile of the absolute difference over wet cells, threshold 2 cm). For the modelling period: run one long event (e.g. `after = 10 d`) per test delta and read the distribution of "time of max water level minus surge peak" over wet cells; the §2.2 `after` must cover its 99th percentile. Define the output table format now so E1–E5 and E6 are directly comparable.

### Acceptance criteria for the chosen method

1. Reproduces the Ebro lag within a factor 2.
2. Agrees with observed celerity within a factor 2 on at least two long backwater deltas (Mississippi, one other).
3. Makes the same materiality decision as the observations in ≥ 90 % of benchmark cases.
4. Produces a defined value (no NaN, no absurd multi-week lags) for every delta in the screening set.
5. Each value carries the regime diagnostics (`k*`, `N`, `M`, drowned fraction, overbank flag) so a reviewer can see why a delta got the lag it got.
6. For the modelling period: on every delta with an existing SFINCS event run, the time of maximum water level at all observation points falls inside the rule's window, and `event_hours` does not exceed the current default by more than a factor the run-time budget allows (to be set; proposal 1.5 on the Σ over deltas).

---

## 5. Proposed decision rule (hypothesis, to be confirmed by E1–E5)

```
for each basin, RP:
    build path table (§3)
    for each reach i (upstream → mouth):
        if z_bed_i < MSL or u_tide_i > u_river_i:      regime = "tidal/drowned";  c_i = sqrt(g*h_rp_i)
        elif dist_i < Lb:                               regime = "backwater";      c_i = 1.5 * u_rp_i (surface slope)
        else:                                           regime = "kinematic";      c_i = 5/3 * u_rp_i (bed slope)
        if Q_i > Q_bf_i: c_i /= f_overbank            # 2–5, default 3
    T_wave = sum(L_i / c_i)
    k_star = Lb / (c_mean * T_event)
    if k_star >= 1/3 and all reaches tidal/drowned:   T_lag = T_wave (quasi-static, hours)
    else:                                             T_lag = peak-to-peak lag from Hayami routing with c_i, D_i   # M6
    tau95 = time before peak while Q >= 0.95 Q_peak
    lag_hr = 0 if T_lag <= tau95 else T_lag
    # modelling period (§2.2)
    T_arrive, T_transit_P90 = from Hayami routing (slow parameter set)
    T_fill = V_fp / (Q_rp - Q_bf);  T_overbank_P90 = L_fp / 0.25
    before = max(2*T_tide, T_surge_half_pre, T_lag + tau_rise)
    after  = max(T_surge_half_post, T_transit_P90 - T_lag, T_fill, T_overbank_P90, 24 h)
    write {basin}_timing.json:
        lag_hr, T_lag, tau95, regime per reach, k_star, N, M, P10/P90,
        window_before_hr, window_after_hr, controlling_term_before, controlling_term_after,
        T_adjust_P90 (for spin-up first guess and calibration run length)
```

`14_build_sfincs.py` reads `lag_hr`, `window_before_hr` and `window_after_hr` from the basin file; the global `sfincs.boundary_setup.compound.lag_hr` (now a fixed shift for every basin, default 0), `window_overrides_hr` and the symmetric `hydrograph.window_hr` become fallbacks for basins without a timing file. The window is used where the surge shapes are built (rule 07c `storm_tide_hydrographs`), so a per-basin asymmetric window reruns rule 07c onward for that basin. The river forcing starts at `−(window_before_hr + lag_hr)` so the shifted rising limb is complete; the blend from the spin-up mean into the event (`max(min_ramp_hr, gap / (max_rise_per_hr × peak))`, capped at the lead-up, `src.river_forcing`) then sits before the shifted limb. The event `tstart` is taken from the spin-up restart file's timestamp, not from `spinup_days` (`14_build_sfincs.py` sets `tstart = tref + spinup_days`), which removes the hidden `spinup_days == lead_days` coupling. Rule 11 (dike-crest calibration) reads `T_adjust_P90` for its run length and storm-tide placement.

---

## 6. Code layout

Test phase (E1–E5), diagnostics only, following the repo's `analysis/` pattern:

```
analysis/
  _upstream_profile.py      # §3 steps 1–3 (exists)
  estimate_peak_travel.py   # 1D river and sea legs, lag, materiality, run length (exists: M1, M2, reduced M4); add M5, M6, T_adjust here
  check_event_timing.py     # SFINCS cross-check on finished runs: time of max, late-filling compartments (exists); add the obs-point lag from river-only runs
tests/unit/test_peak_travel.py    # synthetic channel and sea grid (exists)
tests/unit/test_event_timing.py   # synthetic compartments (exists)
```

Parameters (n range, overbank factor, materiality threshold 0.95, k* threshold 1/3, `after_min` 24 h) go in `config/config.yml` under `testing.flood_timing`, next to `testing.upstream_boundary_check` (protected file: needs a go-ahead). Slope floor and Manning n come from the existing keys.

Pipeline phase, once the rule is accepted: the helpers move to `workflow/src/flood_timing.py` and a rule after rule 11 writes `<results_dir>/{basin}/timing/{basin}_timing.json`, which rules 07c, 11 and 14 read.

---

## 7. Open items to settle before coding

1. What fills the late-peaking compartments (overtopping of calibrated crests, inflow through gaps, or rain-free ponding of the river overflow), and whether they belong in the hazard result at their run-end level or need a longer `after`. Settled by E1.
2. Whether the Ebro's 168 h override can return to the default once the §2.2 rule is in place (p99 time of max +2 h).
3. Tidal velocity amplitude source: GTSM range at the mouth with analytical damping (Cai–Savenije) vs a simple `a·√(g/h)` without damping. The first is better in long tidal rivers (Kapuas-type), the second is enough for short drowned deltas.
4. Peak definition at the coast: discharge peak (M6 output) vs water-level peak. For the compound timing the discharge peak is the right target because the surge sets the water level; record both where observations give both.
5. E2 starts with the six basins that have finished event runs (§0.1); every basin with a rule 11 node table can join for the parametric terms.
6. What the hazard output is: if the product is the maximum water depth only, the window ends once the max has occurred everywhere (the §2.2 rule). If flood duration or recession is also an output, `after` has to cover the drainage, which is a different and much longer timescale (floodplain drainage through crevasses and levee gaps) and should be sized separately.

---

## 8. References

- Allen, G.H., David, C.H., Andreadis, K.M., Hossain, F., Famiglietti, J.S. (2018). Global estimates of river flow wave travel times and implications for low-latency satellite data. *GRL* 45, 7551–7560. doi:10.1029/2018GL077914. Data: doi:10.5281/zenodo.1015799; code: doi:10.5281/zenodo.1219784.
- Cai, H., Savenije, H.H.G., Toffolon, M. (2012). A new analytical framework for assessing the effect of sea-level rise and dredging on tidal damping in estuaries. *JGR Oceans* 117, C09023. doi:10.1029/2012JC008000.
- Cai, H., Savenije, H.H.G., Toffolon, M. (2014). Linking the river to the estuary: influence of river discharge on tidal damping. *HESS* 18, 287–304. doi:10.5194/hess-18-287-2014.
- Chung, W.H., Aldama, A.A., Smith, J.A. (1993). On the effects of downstream boundary conditions on diffusive flood routing. *Adv. Water Resour.* 16, 259–275.
- Couasnon, A., Eilander, D., Muis, S., Veldkamp, T.I.E., Haigh, I.D., Wahl, T., Winsemius, H.C., Ward, P.J. (2020). Measuring compound flood potential from river discharge and storm surge extremes at the global scale. *NHESS* 20, 489–504. doi:10.5194/nhess-20-489-2020.
- Dykstra, S.L., Dzwonkowski, B. (2020). The propagation of fluvial flood waves through a backwater-estuarine environment. *WRR* 56, e2019WR025743. doi:10.1029/2019WR025743.
- Eilander, D., Couasnon, A., Ikeuchi, H., Muis, S., Yamazaki, D., Winsemius, H.C., Ward, P.J. (2020). The effect of surge on riverine flood hazard and impact in deltas globally. *ERL* 15, 104007. doi:10.1088/1748-9326/ab8ca6.
- Eilander, D., Couasnon, A., Leijnse, T., Ikeuchi, H., Yamazaki, D., Muis, S., Dullaart, J., Haag, A., Winsemius, H.C., Ward, P.J. (2023). A globally applicable framework for compound flood hazard modeling. *NHESS* 23, 823–846. doi:10.5194/nhess-23-823-2023.
- Hoitink, A.J.F., Jay, D.A. (2016). Tidal river dynamics: implications for deltas. *Rev. Geophys.* 54, 240–272. doi:10.1002/2015RG000507.
- Jobson, H.E. (2000). Estimating the variation of travel time in rivers by use of wave speed and hydraulic characteristics. USGS WRI 00-4187. doi:10.3133/wri004187.
- Jobson, H.E. (2001). Predicting river travel time from hydraulic characteristics. *J. Hydraul. Eng.* 127, 911–918.
- Kästner, K., Hoitink, A.J.F., Torfs, P.J.J.F., Deleersnijder, E., Ningsih, N.S. (2019). Propagation of tides along a river with a sloping bed. *J. Fluid Mech.* 872, 39–73. doi:10.1017/jfm.2019.331.
- Knight, D.W. (2000). Derivation of routing parameters from cross-section survey. HR Wallingford SR 568.
- Lamb, M.P., Nittrouer, J.A., Mohrig, D., Shaw, J. (2012). Backwater and river plume controls on scour upstream of river mouths. *JGR Earth Surface* 117, F01002. doi:10.1029/2011JF002079.
- Lighthill, M.J., Whitham, G.B. (1955). On kinematic waves. I. Flood movement in long rivers. *Proc. R. Soc. A* 229, 281–316.
- Nittrouer, J.A., Shaw, J., Lamb, M.P., Mohrig, D. (2012). Spatial and temporal trends for water-flow velocity and bed-material sediment transport in the lower Mississippi River. *GSA Bull.* 124, 400–414.
- Paola, C., Mohrig, D. (1996). Palaeohydraulics revisited: palaeoslope estimation in coarse-grained braided rivers. *Basin Res.* 8, 243–254.
- Ponce, V.M., Simons, D.B. (1977). Shallow wave propagation in open channel flow. *J. Hydraul. Div.* 103, 1461–1476.
- Ponce, V.M., Li, R.-M., Simons, D.B. (1978). Applicability of kinematic and diffusion models. *J. Hydraul. Div.* 104, 353–360.
- Ponce, V.M. (1989). Engineering Hydrology: Principles and Practices. Prentice Hall. (Criteria N ≥ 85, M ≥ 15.)
- Sriwongsitanon, N., Ball, J.E., Cordery, I. (1998). An investigation of the relationship between the flood wave speed and parameters in runoff-routing models. *Hydrol. Sci. J.* 43, 197–213.
- Tang, X., Knight, D.W., Samuels, P.G. (2001). Wave speed–discharge relationship from cross-section survey. *Proc. ICE Water & Maritime Eng.* 148, 81–96.
- Tsai, C.W. (2003). Applicability of kinematic, noninertia, and quasi-steady dynamic wave models to unsteady flow routing. *Adv. Water Resour.* 26, 1119–1131.
- Tsai, C.W. (2005). Flood routing in mild-sloped rivers — wave characteristics and downstream backwater effect. *J. Hydrol.* 308, 151–167. doi:10.1016/j.jhydrol.2004.10.027.
- USGS Office of Surface Water (1992). Technical memorandum 92.09: adjustment of discharge measurements made at a distance from the gaging station during periods of changing stage and discharge.
- Ward, P.J., Couasnon, A., Eilander, D., Haigh, I.D., Hendry, A., Muis, S., Veldkamp, T.I.E., Winsemius, H.C., Wahl, T. (2018). Dependence between high sea-level and high river discharge increases flood hazard in global deltas and estuaries. *ERL* 13, 084012. doi:10.1088/1748-9326/aad400.
- Wong, T.H.F., Laurenson, E.M. (1983). Wave speed–discharge relations in natural channels. *WRR* 19, 701–706.
