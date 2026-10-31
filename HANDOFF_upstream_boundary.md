# Handoff: upstream boundary check (v2, 2026-10-27)

Status: phases 1–4 done (`analysis/_upstream_profile.py`,
`analysis/check_upstream_boundary.py`, `analysis/benchmark_upstream_boundary.py`,
`testing.upstream_boundary_check`); 4.4 not needed (section 8). Open: Box 1 /
Figure 1 of the paper, and `L_surge` (section 11). This document is the work order for
Claude Code. Phases 1–3 touch only `analysis/` and `tests/`; `config.yml`,
`docs/methods.md` and the paper are touched in phase 4 after the decisions in
section 8.

Code style: minimal docstrings and comments, no change history in code or docs.

## 1. The question

Each delta model's upstream river boundary (the GloFAS forcing points where
the river enters the domain polygon) must lie outside the sea's influence, so
the compound flood analysis does not impose river discharge inside the reach
the surge already controls.

Reviewer comments on the paper text ("sanity check ... using a kinematic
distance estimation and attenuation distance (Box 1, Figure 1)"):

- **DE1:** the kinematic wave approximation does not apply to surge waves
  travelling upstream.
- **DE2:** use the tidal river and estuary literature (Savenije, Hoitink).
  For high-gradient rivers a maximum elevation threshold of about
  10 m + MSL should be sufficient, at least in the current climate.

The check must use the sea's extreme still water level and show that the
boundary is far from where that level influences the river.

## 2. Current implementation (to be replaced)

`analysis/check_upstream_boundary.py [basin_id]` reads
`<results_dir>/<basin_id>/preprocessing_inputs/` and writes
`figs/upstream_boundary_check/<basin_id>_upstream_boundary_check.png`.

- Mouths: reaches whose downstream ids are absent from the network. Forcing
  points: `has_glofas=1` crossings in `river_forcing.nc`.
- `A0`: `storm_tide_at_rp_m` of the mouth's nearest station at
  `testing.upstream_boundary_check.surge_rp_yr`.
- Distance = `min(kinematic, attenuation)` as a geodesic radius around each
  mouth; forcing points inside a radius are flagged.
- Config keys: `effective_period_fraction`, `amplitude_threshold_fraction`,
  `surge_rp_yr`.

## 3. Why each condition is kept, replaced or demoted

- **"Kinematic" distance `T_eff (c − v)`, `c = √(g(h + A0))`.** Dropped.
  `√(gh)` is the dynamic celerity, not kinematic, and the formula is a
  frictionless travel distance over an arbitrary fraction of the period. With
  no friction there is no physical limit to how far a long wave travels, so
  the result is set entirely by `effective_period_fraction`. The same formula
  must be checked wherever `L_surge` is computed for domain delineation
  (`max(3·L_b, L_surge)`), because DE1 applies there identically.
- **Attenuation march `A = A0 exp(−μx)`.** Replaced, not dropped. It is a
  stripped-down tidal damping balance without channel convergence (amplifies)
  and river discharge (damps). The full balance is step 4.4.
- **Still-water intersection `x_SWL`.** Kept as a diagnostic, not as the
  criterion. It is the intersection of a horizontal `SWL_ext` with the
  undisturbed normal-depth surface: a lower bound of sea influence. The M1
  backwater profile lies above both lines and reaches normal depth only
  asymptotically. It is adequate for steep rivers because the backwater tail
  is short there; the Ebro caveat in section 9 is the general low-gradient
  case.
- **Backwater length `h/S`.** Kept as a diagnostic. Linearised GVF (wide
  channel, Manning, small perturbation `Δ₀` at the mouth) gives
  `Δ(x) ≈ Δ₀ exp(−3 S x / h)`; `h/S` is the ~5 % residual length. Fails when
  `Δ₀/h` is not small, when `S → 0` (lowest delta reaches), and it is a steady
  result (conservative envelope for a 1–3 day pulse).
- **Steady GVF backwater profile.** New core. `x_SWL` and `h/S` are both
  approximations to this profile; integrating it directly handles large
  `Δ₀/h`, zero-slope reaches and arbitrary (Q, SWL) pairs. The bed exists
  since the river depth is set (rule 10/11 or the calibration); width and
  Manning's n exist.
- **10 m + MSL rule.** Demoted to a flagged fallback. It works for steep
  rivers because steep means short `h/S`, not because 10 m is special; with
  ~7 m storm tides plus ~1 m SLR the margin is ~2 m. Used only when the GVF
  cannot run and the slope exceeds a threshold.
- **Savenije/Cai analytical damping.** Optional, conditioned. The only
  condition that represents convergence amplification, friction, inertia and
  river discharge (Horrevoets et al. 2004; Savenije et al. 2008; Cai et al.
  2014). Assumes exponential width convergence, uniform depth, periodic
  small-amplitude wave; for a surge the "period" is the pulse duration, which
  makes it an extrapolation. Relevant where the convergence length is short
  relative to `h/S`; the Ebro will not need it. Framing reference for Box 1:
  Hoitink & Jay (2016).

## 4. Method (develop mode; `analysis/` only)

### 4.1 Inputs per basin

From `<results_dir>/<basin_id>/preprocessing_inputs/`:

- rule 11's round-0 node network and bed (`src.river_bed`), rebuilt with the
  current code: rule 10's surface (5 km smoothed FathomDEM, mouths at the calm
  sea level), the bed = surface − Manning depth at bankfull discharge on the
  40 km slope, and the node widths. This is the bed the river burn uses;
- mainstem path mouth → each forcing point along that network (largest
  bankfull discharge at bifurcations), river distance from the node chainage;
- channel Manning's n (`river_processing.channel_manning_n`);
- discharges: `Q_mean` and `Q_bf`; no separate low flow (`Q_low = Q_mean`);
- sea levels at the mouth's nearest station from `surge_forcing.nc`
  (EGM2008, MDT-corrected): `station_calm_sea_level_m`, `storm_tide_at_rp_m`
  at `surge_rp_yr = 100`, and `SWL_ext = storm_tide_at_rp_m + 0.5 m ×
  slr_fingerprint` (SLR fixed at 0.5 m, the largest scenario);
- surge hydrograph duration (only for 4.4).

### 4.2 Core: steady GVF profile

Steady GVF over the whole node network (`src.gvf_solver.forward_network`, the
solver of rule 11's forward check): rectangular section of the node width,
Manning friction, subcritical (Froude capped), each reach integrated upstream
from the level at its downstream end, branches included. Sea mouths take the
level of the nearest surge station; other network ends keep rule 10's level.
A mouth bed above the sea level starts at the bed. Zero-slope reaches need no
special handling.

```
S_f = n² Q² / (A² R^(4/3)),  dh/dx = (S0 − S_f) / (1 − Fr²)
```

Run the pair matrix `Q ∈ {Q_mean, Q_bf}` × `SWL ∈ {calm, storm tide at
RP100, SWL_ext}`. Mean discharge is the reference pair for the verdict, because
the surge can precede the river peak by days.

### 4.3 Diagnostics from the profiles

Per forcing point (river distance `x_f`):

- `ΔWL(x) = WL(Q, SWL_ext) − WL(Q, calm)`;
- `x_inf`: first node where `ΔWL < max(level_tol_m, amplitude_threshold_fraction × A0)` (influence length);
- `ΔWL(x_f)`: perturbation at the forcing point;
- `L_fit`: e-folding length from a fit of `ΔWL(x) = A0 exp(−x/L)` over
  `x < x_inf`;
- `x_SWL`: first node where `WL(Q_mean, calm) > SWL_ext` (rule 10's surface
  runs metres above the mean-flow level on steep rivers: 2209166 x_SWL 4.9 km
  on the surface, ~14 km on the mean-flow level, SFINCS influence 13.8 km);
- `L_b = h/S` with `h` the derived depth at the mouth and the mean bed slope
  mouth → forcing point (reported for comparison with `L_fit`, not used in
  the verdict);
- height of the river surface above `SWL_ext` and above local MSL at `x_f`.

The bed is cut for bankfull discharge on the 40 km slope under the 5 km
surface, so `WL(Q_bf, calm)` departs from rule 10's surface by design (Ebro:
−0.4 to +0.9 m, rule 11's forward check). `ΔWL` takes both levels from the
same bed and discharge, so that offset cancels; it is reported at the forcing
point (`forward_minus_surface_m`) for reference only. `x_SWL`, the heights
above `SWL_ext` and MSL and the 10 m rule use `WL(Q_mean, calm)`.

### 4.4 Optional: analytical damping (implement only if phase 3 asks for it)

Cai et al. (2014, JGR) hybrid solution with river discharge; convergence
length from an exponential fit to SWORD widths along the mainstem; depth from
`river_depth_m`; period = surge pulse duration from the hydrograph. Output:
distance where amplitude falls below `amplitude_threshold_fraction × A0`, plus
a flag for amplification (amplitude increasing upstream). Trigger: convergence
length < `L_b` or tidal range at the mouth > 1 m.

### 4.5 Fallback: 10 m rule

Only when 4.2 cannot run (no depth) and the mean bed slope > `high_gradient_slope_m_per_km`.
Pass if the river surface at `x_f` ≥ `high_gradient_elevation_m` above local
MSL; within 1 m of the threshold → "marginal", not "pass".

### 4.6 Verdict

Pass when both hold at `Q_mean`:

1. `ΔWL(x_f) < level_tol_m`
2. `x_f ≥ x_inf + margin_km`

If 4.4 ran, replace `x_inf` by `max(x_inf, damping distance)`. Report the
deciding criterion and, for fails, the distance and height still missing.

### 4.7 Outputs

- `figs/upstream_boundary_check/<basin_id>_forcing_points.csv`: one row per
  forcing point × (Q, SWL) pair with all quantities in 4.3, verdict, deciding
  criterion, depth source, surface source.
- `figs/upstream_boundary_check/<basin_id>_profile_<mouth>.png`: river
  distance vs elevation — bed, surface at each pair, `SWL_ext`, calm level,
  10 m line, `x_SWL`, `x_inf`, forcing points.
- The existing map with along-river markers (`x_inf`, `x_SWL`, forcing
  points) instead of circles.
- `--extra-reach` evaluates a boundary at the upstream end of other reaches
  (the old Ebro boundary, …041). The v1 distances are not kept; the
  before/after numbers on the Ebro are in the CHANGELOG (v1 radius 65.3 km,
  forcing point inside).

## 5. Test plan

### 5.1 Unit tests (synthetic channel, `tests/test_upstream_boundary.py`)

- Uniform slope, constant width, Manning: the GVF profile matches the
  analytical M1 solution; the fitted decay length equals `h/(3S)` within 10 %;
  `x_SWL = SWL_ext / S` within one node; `L_b = h/S`.
- Flat lowest reaches then a sloping reach (Ebro-like): no division by zero;
  `x_inf` lies on the sloping reach.
- `Δ₀/h` large (surge of 5 m on 6 m depth): march stays subcritical and
  `x_inf` exceeds the linearised estimate.
- Forcing point at exactly `x_inf`: fails with margin > 0, passes with margin 0.
- Mouth bed above sea level: handled, flagged.
- If 4.4 is implemented: damping distance decreases with discharge and with
  friction; no damping when convergence balances friction.

### 5.2 Diagnostic on the Ebro (2433835), no SFINCS

Run after `snakemake preprocess --config target_basins="[2433835]"` on the
revised polygon, and once on the old outputs for the before/after table.
Expected (section 9): old boundary fails, new boundary passes at `Q_mean`; the
interesting number is `ΔWL(x_f)` at the new boundary, which `L_b` cannot give.

Sensitivity sweeps on the same basin:

- surface source: rule 10 vs SWORD vs SWOT (1.6 m apart at the new crossing);
- `Q_mean` vs `Q_bf`;
- SLR: 0.5 m vs the 2050 scenario vs none (RP fixed at 100);
- `level_tol_m` ∈ {0.02, 0.05, 0.10}, `margin_km` ∈ {2, 5, 10}.

Report how far `x_inf` and the verdict move.

### 5.3 Ground truth: paired SFINCS runs

The formulas cannot validate each other; the hydrodynamic model is the only
reference. Per test basin, two steady runs on the current polygon with
identical river forcing (`Q_mean`, steady):

- **calm**: `station_calm_sea_level_m` at the sea boundary;
- **surge**: `SWL_ext` held steady (and, if cheap, the RP hydrograph as a
  third run to measure the transient reduction).

Extract water level at the mainstem SWORD nodes from both; take `ΔWL_sfincs(x)`,
`x_obs` (same threshold as `x_inf`) and `L_obs`. Compare with 4.3.

Basins (one per class):

1. low gradient, small surge: Ebro, old and revised polygon; if the existing
   Ebro run saved water levels, use it first to see whether the surge reached
   the old 47 km boundary;
2. low gradient, large surge: a delta with `storm_tide_at_rp_m` of several
   metres (pick from the station table);
3. high gradient: a steep delta (pick by mean bed slope).

The SFINCS runs share the GVF bed, so they test the 2D and transient
behaviour, not the bed.

### 5.4 Decision rule

- Steady GVF is accepted as the criterion if `x_inf ≥ x_obs` in all three
  basins and `x_inf / x_obs` is not absurd (< ~3).
- 4.4 is implemented if, in the large-surge basin, `L_obs` is much longer
  than `L_fit` or amplification is seen in SFINCS.
- The 10 m rule stays as fallback only if `x_obs` in the steep basin lies
  well below the 10 m contour.
- `level_tol_m` and `margin_km` are set from the spread between `x_inf` and
  `x_obs`, not by hand.

## 6. Phases (tasks for Claude Code)

1. **Profile extraction.** One function that returns the mainstem path mouth →
   forcing point with distance, width, slope, surface, bed, n, and one that
   samples a SFINCS output at the same nodes. Used by 4.2 and 5.3.
2. **Diagnostic.** 4.2–4.3, 4.5–4.7; parameters read from a
   `develop` block with defaults, `config.yml` untouched. Unit tests 5.1.
   Run 5.2 and send the CSV and profiles.
3. **Benchmark.** 5.3 on the three basins; table of `x_inf`, `x_obs`, `L_fit`,
   `L_obs`, verdicts. Stop and report before 4.4.
4. **Settle.** Decisions in section 8 → `config.yml`,
   `docs/methods.md`, note for Box 1 / Figure 1, fix `L_surge` in the domain
   delineation.

## 7. Config change (phase 4 only)

`testing.upstream_boundary_check`:

- remove `effective_period_fraction`;
- keep `amplitude_threshold_fraction`; `surge_rp_yr: 100`;
- add `slr_m: 0.5`, `level_tol_m`, `margin_km`,
  `high_gradient_elevation_m: 10`, `high_gradient_slope_m_per_km`,
  `use_analytical_damping: false`.

## 8. Decisions

Settled:

- `SWL_ext`: the most extreme case, per station the highest storm tide + SLR ×
  fingerprint over the scenarios in `scenarios.yml` with a surge RP and a
  stress case (`stress_surge_rp_yr` 100, `stress_slr_m` 1.0), which sets it
  in all three benchmark basins. The table below is at RP100 + 0.5 m SLR.
- No separate low flow: `Q_low = Q_mean`
- Bed: from the pipeline DEM (GVF depth estimate at `Q_mean` burnt in)

- Benchmark basins: 2433835 (low gradient, small surge), 3279946 (low
  gradient, 4 m surge), 2209166 (steep, 2.0 m/km); steady runs of 4–8 days
- Steady GVF is the criterion (5.4):

  | basin, reach | x_f | x_inf (GVF) | x_obs (SFINCS) | x_inf/x_obs | L_fit | L_obs | 10 m contour |
  |---|---|---|---|---|---|---|---|
  | 2433835, …071 | 65.5 | 54.3 | 55.7 | 0.97 | 109 | 80 | 64.0 |
  | 2209166, …061 | 42.1 | 13.8 | 13.8 | 1.00 | 10 | 13 | 18.3 |
  | 3279946, …155 | 264.5 | 198.3 | 198.3 | 1.00 | 914 | 358 | 246.7 |
  | 3279946, …031 | 293.6 | 275.9 | 276.9 | 1.00 | 361 | 323 | 277.4 |

  (km; per forcing point's mainstem). Along all main arms
  (`figs/upstream_boundary_benchmark/README.md`) x_inf falls short of x_obs
  by at most 2.9 km (3279946, western arm, 322 km). SFINCS level rise decreases monotonically upstream (no
  amplification) and `L_obs ≤ 1.2 L_fit`, so 4.4 is not implemented. The e-folding fit is a poor summary of
  the profiles (flat backwater tail, then a sharp drop where the river
  steepens) and stays a diagnostic only.
- `level_tol_m = 0.05`, `margin_km = 5`: the threshold was set by the
  fraction term (0.05–0.20 m) in all basins; `margin_km` covers ~3× the
  largest shortfall of x_inf on the forcing mainstems and 1.7× the largest
  on any arm.
- 10 m rule: reported for comparison only (every built basin has a bed, so
  the 4.5 fallback never applies; no slope threshold kept). The steep basin's x_obs (13.8 km) lies 4.5 km below the 10 m contour; on the
  low-gradient 4 m-surge basin the contour lies only 0.5 km beyond x_obs.
- Box 1 / Figure 1 note: replace the kinematic distance and the attenuation
  march by the steady GVF level rise (calm vs SWL_ext, mean discharge),
  checked against steady SFINCS in three basins (table above); x_SWL and h/S
  as approximations; 10 m + MSL (DE2) compared, not used.

## 9. Ebro check on the revised polygon (manual, raw data; reference)

Revised polygon: `<raw_data_root>/DeltaWebs/modified/9_polygons.gpkg`
(BasinID2 2433835), +221 km² (1012 → 1233 km²), domain extended 13.5 km north.
Pipeline outputs still reflect the old polygon.

Extreme still water level at the mouth (9 stations within 0.02 m). The manual
check used RP250 and 0.3 m SLR; the settled values are RP100 and 0.5 m SLR,
so `SWL_ext` rises to about 1.01 m with the RP250 storm tide and somewhat less
with RP100. The margins below shrink by ~0.2 m but the verdicts do not change;
the diagnostic recomputes them.

| | level (EGM2008) |
|---|---|
| `storm_tide_at_rp_m` (RP250, current `surge_forcing.nc`) | 0.44 m |
| plus 0.3 m × fingerprint 1.14 (manual check) | **0.78 m** |
| plus 0.5 m × fingerprint 1.14 (settled) | **1.01 m** |
| MDT correction (local MSL to geoid) | −0.05 m |

One river crossing on both polygons (Ebro main stem):

| | old boundary | new boundary |
|---|---|---|
| SWORD reach | 21604100041 | 21604100071 |
| distance from mouth (`dist_out`) | ~47 km | ~78 km |
| crossing (EPSG:32631) | 289219, 4518147 | 290565, 4538195 |
| SWORD `wse` (EGM96) | 0.8 m | 9.2 m |
| SWOT `wse_obs_p50` | 1.2 m | 10.8 m |
| FathomDEM at crossing | 0.0 m | 12.0 m |
| margin above 0.78 m | ≈ 0–0.4 m | ≈ 8.4–11.2 m |
| 10 m + MSL rule | fails | met by SWOT/FathomDEM; SWORD 0.8 m short |

FathomDEM at the new crossing: levelled in 0.5 m steps along the centreline
(11.5 / 12.0 / 12.5 m); a flat 12.0 m band 75–100 m wide on the SWORD line,
banks 13–21 m, valley wall to 80 m; one 47 m sample at −1.4 km ignored.

Interpretation: old boundary inside the still-water reach; new boundary ~30 km
upstream of `x_SWL` (≈ 47 km at 0.78 m; a little further at 1.01 m) and
8.2–11.0 m above the settled `SWL_ext`. Caveat: `h/S` ≈ 10 m / 0.12 m km⁻¹ ≈
85 km, comparable to the boundary distance, so the level at the boundary may
still respond weakly at mean discharge; the linearised estimate
`1.01 × exp(−3 × 78/85)` ≈ 0.06 m is what 4.2 and 5.3 must confirm — right
at the proposed `level_tol_m = 0.05`, which is why the GVF profile and the
SFINCS benchmark decide this rather than the scaling.

## 10. Data notes

- Datums: FathomDEM EGM2008 in int32 cm (÷100); SWOT EGM2008; SWORD `wse`
  MERIT on EGM96 (small difference at these margins, but convert).
- `storm_tide_rp_table` holds raw COAST-RP levels (local MSL);
  `storm_tide_at_rp_m` and `station_calm_sea_level_m` include the MDT
  correction.
- SWORD `slope` and `wse` are 0 on the lowest Ebro reaches (`dist_out` < 36 km).
  The GVF march handles this; `L_b` uses the mean slope mouth → boundary.

## 11. Related

- Domain delineation `max(3·L_b, L_surge)`: if `L_surge` uses the v1
  "kinematic" formula, DE1 applies there too. It is not computed in this
  repo (the domain polygons are an input), so the fix belongs where the
  polygons are made.
- B27 in `TODO.md`: hydromt `align=True` can leave the grid up to half a cell
  short of the polygon; negligible here.
- `docs/methods.md` holds the surge and river forcing notes.
- Box 1 and Figure 1 of the paper are rewritten once section 8 is settled;
  Hoitink & Jay (2016) as framing, Savenije (2012) and Cai et al. (2014) for
  the damping formulation if 4.4 is used.
