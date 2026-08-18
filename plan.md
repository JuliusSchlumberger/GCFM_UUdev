# Adaptation-strategy extension: `pre` vs `post` method

## Context

The `adaptation_modelling` branch already has a "menu" of adaptation measures (`config/measures.yml`), a set of named "strategies" composing specific measures at specific parameter values (`config/adaptation_strategies.yml`), and a fully-written measure-application library for one of two methods (`workflow/src/adaptation_method_pre.py`, which mutates a `hydromt_sfincs` model via `dispatch_rules()`). What's missing is the Snakemake wiring that actually invokes this, plus a second method that manipulates the already-computed flood map instead of rerunning SFINCS.

The user wants to compare two ways of applying the same strategy to the same baseline scenario run:
- **`pre`**: apply measures to the SFINCS model → rebuild forcing → rerun SFINCS → recompute metrics. Physically correct, expensive.
- **`post`**: apply measures directly to the baseline scenario's already-computed `max_flood_depth.tif` → recompute metrics. Cheap, approximate.

They are new to Snakemake and explicitly want the existing pipeline (rules 00–17) left alone, and the new work to reuse existing code/patterns rather than duplicate logic. This plan was produced after two research passes over the exact rule/script signatures involved (`00_common.smk`, `Snakefile`, rules 13/14/16/17 and their scripts, `src/postprocessing.py`, `src/sfincs_run.py`, `adaptation_method_pre.py`, `config/measures.yml`, `config/adaptation_strategies.yml`, `config/config.yml`) and directly verified against the current file contents — no material discrepancies found. Two design calls were confirmed with the user: implement post-method placeholder formulas for all 6 measure types now (documented as approximations), and include a method-agnostic comparison rule in this pass.

**2026-08-13 revision**: the `post` method's design (sections 4/5 below) was superseded once implementation started. Instead of the original in-memory bathtub-truncation approximation, `post` now reuses a more physically-detailed, per-source (river/coastal/compound) measure library ported from the user's own sibling `delta_model` project, driven by a per-basin×scenario `attribution_mask.tif` classifying each pixel's dominant flood source. This needed one new upstream rule (attribution-mask generation, section 4a) plus a new counterpart-scenario lookup — it still requires **no new SFINCS runs**: the attribution mask is built entirely from already-existing single-driver sibling scenarios (`coast_only`/`river_only`/`coast_only_500`/`river_only_500`, plus `default` as the "mean conditions" baseline class) that `config/scenarios.yml`'s own `# Attribution runs` comment had already anticipated. Also renamed during implementation: the `offshore_island` measure is now `offshore_barrier` everywhere (`config/measures.yml`, `config/adaptation_strategies.yml`) — `adaptation_method_pre.py`'s own `selected_measures` dict has **not** been updated to match yet, see Open items.

## Design summary

- New axes `STRATEGIES` (from `adaptation_strategies.yml` top-level keys, excluding `meta`) and `METHODS` (`"pre"`/`"post"`, both by default) are added to `workflow/rules/00_common.smk`, mirroring how `SCENARIOS`/`SCENARIO_DEFS` already work. `STRATEGIES` defaults to **empty** — adaptation is opt-in only via `--config target_strategies=[...]`, exactly as `config.yml`'s existing forward-looking comment already promises.
- `{strategy}` **is** a real Snakemake wildcard (regex-constrained, exactly like `{scenario}` — strategy names use the same `[A-Za-z0-9_-]+` convention, no dots, e.g. `advance_0_9` not `advance_0.9`).
- `{method}` is **never** a matched Snakemake wildcard — it only ever appears as a literal `"pre"`/`"post"` path segment, because no rule body is shared between the two methods (one is SFINCS-model-based, the other is raster-based). Two rules matching the same output pattern for the same wildcard values is the ambiguity this avoids. `METHODS`/`target_methods` still exist as a config axis driving which literal-path outputs get requested.
- **`pre` chain**: reuses the existing skeleton→build→run→metrics shape. One new script (`18a_adapt_pre.py`) plays the role of `build_sfincs_skeleton` — it loads the basin's skeleton read-only, redirects the root, and applies the strategy's measures via `dispatch_rules()` from the untouched `adaptation_method_pre.py`. Downstream of that, `13_build_sfincs.py`, `16_run_event.py`, and `17_flood_metrics.py` are **reused completely unmodified** — new rules just point their `params.skeleton_root`/`params.sfincs_root`/`output:` at adaptation paths.
- Basin-level spin-up (rule 14) is **never rerun per strategy**. None of the implemented measures touch the coarse grid's bed elevation/active mask (weirs and drainage structures are subgrid-level; `retreat` only changes subgrid roughness, not elevation), so the baseline restart file remains a valid initial condition for every adapted run.
- The apply step (`18a_adapt_pre.py`) is scoped **basin×scenario×strategy**, not basin×strategy, because `retreat` needs the *scenario's own* baseline flood map to determine eligible cells — eligibility genuinely differs by return period. This is applied uniformly (not just for strategies that include `retreat`) to keep one DAG shape and avoid a "which measures need scenario data" registry that could drift out of sync with `adaptation_method_pre.py`.
- **`post` chain** (revised 2026-08-13): one rule (`adapt_metrics_post`) + one script (`18b_adapt_post.py`), fed by one new **upstream** rule (`attribution_mask`, section 4a) that is basin×scenario-scoped (runs once, shared across every strategy applied to that scenario — not once per strategy). `adaptation_method_post.py`'s measures are **file-based**: each one reads a `flood_map_path` raster + `scenario_root/attribution_mask.tif`, writes a new `max_flood_depth.tif` under `output_dir`, and returns `{"method":..., "out_raster": str}` — chained by threading one measure's `out_raster` into the next measure's own `flood_map_path`. This replaced the originally-planned in-memory `{"da_hmax": xr.DataArray, "landuse_path": str}` state-dict design (never implemented). Reuses `compute_risk_metrics` from `src/postprocessing.py` unmodified on the final chained raster.
- Both chains converge on the same artifact shape: `.../adaptation/<pre|post>/{strategy}/max_flood_depth.tif` + `.../flood_metrics.csv`, both directly in the strategy folder — no `visuals/`/`metrics/` split (unlike the baseline pipeline's own `runs/{scenario}/visuals/`+`metrics/` convention, which is intentionally NOT mirrored here per your request for one flat output folder per strategy). For the `pre` method, the SFINCS model's own working files still need two dedicated sub-folders (`sfincs_skeleton/`, `sfincs/`) — SFINCS requires a file literally named `sfincs.inp` in its working directory, and the "apply measures" step and the "build forcing" step each produce their own distinct `sfincs.inp` that Snakemake must track as two different rules' outputs, so those two can't be merged into the result folder without either duplicating `13_build_sfincs.py`'s logic or refactoring it — out of scope here. Everything that isn't part of the SFINCS engine's own working files (the actual results: flood map, metrics, plots, animation, timeseries) is flattened into one folder.
- A new `rule adapt` aggregates all adaptation outputs, opt-in only. `rule all`, `rule build`, `rule preprocess` are **left byte-for-byte unchanged**.
- A comparison rule concatenates baseline/pre/post metrics CSVs into one long-format CSV with delta columns.

## 1. `workflow/rules/00_common.smk` additions

Append after the existing `SCENARIOS` block, before `wildcard_constraints`:

```python
# ── adaptation strategy & method axes ────────────────────────────────────────
with open(config["adaptation"]["strategies_file"]) as _f:
    STRATEGY_DEFS_RAW = _yaml.safe_load(_f) or {}

ADAPT_CATALOGUE_ROOT = STRATEGY_DEFS_RAW.get("meta", {}).get("root", ".")
STRATEGY_DEFS = {k: v for k, v in STRATEGY_DEFS_RAW.items() if k != "meta"}
if not STRATEGY_DEFS:
    raise ValueError(f"{config['adaptation']['strategies_file']} defines no strategies (besides 'meta')")

for _name in STRATEGY_DEFS:
    if not _re.fullmatch(r"[A-Za-z0-9_-]+", _name):
        raise ValueError(f"strategy name {_name!r} invalid (use letters/digits/_/- only)")

with open(config["adaptation"]["measures_file"]) as _f:
    MEASURES_DEFS = (_yaml.safe_load(_f) or {}).get("adaptation_measures", {})

def adaptation_input_path(rel):
    """Resolve a strategy measure's locations/dep_subgrid string param
    against adaptation_strategies.yml's own meta.root."""
    return str(Path(ADAPT_CATALOGUE_ROOT) / rel)

def strategy_measure_input_paths(strategy):
    """Raw-data files a strategy's measures reference, as extra `input:`
    so editing e.g. adaptation/levee.geojson triggers a rerun."""
    paths = []
    for _measure_params in STRATEGY_DEFS[strategy]["measures"].values():
        for _k, _v in _measure_params.items():
            if _k in ("locations", "dep_subgrid") and isinstance(_v, str):
                paths.append(adaptation_input_path(_v))
    return paths

# Opt-in only -- no default strategy set (unlike scenario's "default"):
#   snakemake adapt --config target_strategies="['retreat']"
STRATEGIES = list(config.get("target_strategies", []))
_unknown = sorted(set(STRATEGIES) - set(STRATEGY_DEFS))
if _unknown:
    raise ValueError(f"target_strategies {_unknown} not defined in {config['adaptation']['strategies_file']}")

# Both methods by default once a strategy is selected:
#   snakemake adapt --config target_strategies="['retreat']" target_methods="['pre']"
METHODS = list(config.get("target_methods", ["pre", "post"]))
_unknown_methods = sorted(set(METHODS) - {"pre", "post"})
if _unknown_methods:
    raise ValueError(f"target_methods {_unknown_methods} invalid -- only 'pre'/'post' supported")
```

Extend the existing `wildcard_constraints:` block (additive only):

```python
wildcard_constraints:
    basin_id = r"\d+",
    scenario = r"|".join(sorted(SCENARIO_DEFS)),
    strategy = r"|".join(sorted(STRATEGY_DEFS)) if STRATEGY_DEFS else r"(?!)",
```

No `method` entry — it's never a matched wildcard (see design summary).

**(2026-08-13 addition)** Also add a counterpart-scenario lookup, alongside `scenario_params` — used by the new `attribution_mask` rule (section 4a):

```python
def attribution_counterparts(scenario):
    """-> (river_only_scenario, coastal_only_scenario): the two sibling
    single-driver scenario names whose own RP matches `scenario`'s own
    river_rp/surge_rp exactly (river_rp=X & surge_rp=None / surge_rp=Y &
    river_rp=None) -- e.g. coast_500 (surge=500, river=2) pairs with
    river_only (river=2) and coast_only_500 (surge=500); river_500
    (surge=2, river=500) pairs with river_only_500 and coast_only (surge=2).
    Raises if scenarios.yml has no such sibling defined yet (e.g. coast_100
    has no coast_only_100 counterpart) -- confirmed with the user rather
    than silently guessing a mismatched RP."""
    river_rp = SCENARIO_DEFS[scenario]["river_rp"]
    surge_rp = SCENARIO_DEFS[scenario]["surge_rp"]

    def _find(rp_key, other_key, rp_val):
        matches = [n for n, s in SCENARIO_DEFS.items()
                   if s.get(rp_key) == rp_val and s.get(other_key) is None]
        if not matches:
            raise ValueError(
                f"No {rp_key}={rp_val}-matched, {other_key}=None sibling scenario "
                f"for {scenario!r} -- add one to config/scenarios.yml"
            )
        return matches[0]

    river_only = _find("river_rp", "surge_rp", river_rp)
    coastal_only = _find("surge_rp", "river_rp", surge_rp)
    return river_only, coastal_only
```

Baseline ("mean conditions", overrides all other classes) is fixed, not RP-matched: always the `"default"` scenario's own `max_flood_depth.tif`.

## 2. Converged output path shape

```
{RESULTS_DIR}/{basin_id}/runs/{scenario}/sfincs/attribution_mask.tif                              # attribution_mask (NEW, 4a)
{RESULTS_DIR}/{basin_id}/runs/{scenario}/sfincs/attribution_mask.png                               # attribution_mask (NEW, 4a)

{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp   # adapt_apply_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp             # adapt_build_forcing_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc          # adapt_run_event_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/01_inundation_ratio.png       # adapt_run_event_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/02_flood_animation.mp4        # adapt_run_event_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_timeseries.csv          # adapt_run_event_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/max_flood_depth.tif            # adapt_flood_metrics_pre
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv              # adapt_flood_metrics_pre

{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif           # adapt_metrics_post
{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv             # adapt_metrics_post

{RESULTS_DIR}/{basin_id}/runs/{scenario}/adaptation/compare/{strategy}/metrics_comparison.csv    # adapt_compare
```

## 3. New rule file: `workflow/rules/18a_adapt_pre.smk` (4 rules)

```python
rule adapt_apply_pre:
    # Applies this strategy's measures (via adaptation_method_pre.dispatch_rules,
    # unmodified) to a COPY of the basin's skeleton -- same read-skeleton/
    # redirect-root pattern as build_sfincs.py. Scoped basin x scenario x
    # strategy because `retreat` needs THIS scenario's own baseline flood map.
    # Never re-runs spin-up (see design summary).
    input:
        skeleton_inp       = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        measure_data       = lambda wildcards: strategy_measure_input_paths(wildcards.strategy),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
    params:
        strategy_def       = lambda wildcards: STRATEGY_DEFS[wildcards.strategy],
        measures_def       = MEASURES_DEFS,
        adaptation_root    = ADAPT_CATALOGUE_ROOT,
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        adapted_root       = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/18a_adapt_pre.log"
    script: "../scripts/18a_adapt_pre.py"


rule adapt_build_forcing_pre:
    # Reuses scripts/13_build_sfincs.py UNMODIFIED -- it never reads
    # snakemake.wildcards, only snakemake.params paths.
    input:
        skeleton_inp    = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        grid_resolution = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
        rstart          = results_path("{basin_id}/spin_up/" + RST_FNAME),   # baseline restart, reused
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
    params:
        # identical to rule build_sfincs's own params block --
        depth_method = config["river_processing"]["depth_method"],
        resolution   = lambda wildcards, input: json.load(open(input.grid_resolution))["resolution"],
        tref = config["sfincs"]["simulation"]["tref"],
        dtmapout = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout = config["sfincs"]["simulation"]["dthisout"],
        storevelmax = config["sfincs"]["simulation"]["storevelmax"],
        storetwet = config["sfincs"]["simulation"]["storetwet"],
        include_rstart = config["sfincs"]["spinup"]["enabled"],
        spinup_days = config["sfincs"]["spinup"]["spinup_days"],
        rst_fname = RST_FNAME,
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        discharge_multiplier = config["boundary_forcings"]["river"]["discharge_multiplier"],
        slr_enabled = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m = config["boundary_forcings"]["surge"]["slr"]["slr_m"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        # only these two differ from build_sfincs's own params:
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        spin_up_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/13_build_sfincs.log"
    script: "../scripts/13_build_sfincs.py"          # REUSED, UNMODIFIED


rule adapt_run_event_pre:
    # Reuses scripts/16_run_event.py UNMODIFIED.
    input:
        sfincs_inp          = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
        rstart              = results_path("{basin_id}/spin_up/" + RST_FNAME),
        land_polygons       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        landuse             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc            = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
        plot_inundation_ratio    = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/01_inundation_ratio.png"),
        animation_flood_progress = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/02_flood_animation.mp4"),
        flood_timeseries_csv     = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_timeseries.csv"),
    params:
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        # the ADAPTED skeleton -- subgrid dep/roughness differs there for `retreat`
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        sfincs_exe = config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
        animation_fps = config["sfincs"]["sanity_checks"]["animation_fps"],
    threads: workflow.cores
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/16_run_event.log"
    script: "../scripts/16_run_event.py"             # REUSED, UNMODIFIED


rule adapt_flood_metrics_pre:
    # Reuses scripts/17_flood_metrics.py UNMODIFIED -- it uses
    # snakemake.wildcards.basin_id/scenario directly, both still correct here.
    input:
        sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
        landuse       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        delta_polygon = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
    params:
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        hmin = config["metrics"]["hmin"], urban_code = config["metrics"]["urban_landuse_code"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/17_flood_metrics.log"
    script: "../scripts/17_flood_metrics.py"         # REUSED, UNMODIFIED
```

### `workflow/scripts/18a_adapt_pre.py` (NEW) — pseudocode

- `sf = SfincsModel(root=params.skeleton_root, mode="r"); sf.read(); sf.root.set(params.adapted_root, mode="w+")` — same idiom as `13_build_sfincs.py`/`14_run_spinup.py`.
- Local mapping `COMPONENT_OF_MEASURE = {"offshore_barrier": "weirs", "coastal_and_river_levee": "weirs", "coastal_levee": "weirs", "dike_ring": "weirs", "pumps": "drainage_structures", "retreat": "subgrid"}`.
- `touched = set()`; loop `for measure_type, raw_params in params.strategy_def["measures"].items()`:
  - `measure_def = params.measures_def[measure_type]`
  - resolve `locations`/`dep_subgrid` string params via `adaptation_input_path(params.adaptation_root, v)`
  - `flood_map_path = input.baseline_flood_map if measure_type == "retreat" else None`
  - `sf = dispatch_rules(measure_type, sf, measure_def, flood_map_path=flood_map_path, method="preprocessing", **resolved_params)` — from `src.adaptation_method_pre`, unmodified
  - `touched.add(COMPONENT_OF_MEASURE[measure_type])`
- Write only touched components (`sf.weirs.write()` / `sf.drainage_structures.write()` / `sf.subgrid.write()`).
- Hand-craft the adapted `sfincs.inp` via `parse_sfincs_inp` + `forward_geometry_files(skeleton_cfg, skeleton_root, adapted_root, exclude=<touched *file keys>)` for untouched geometry files, appending the just-written touched files' bare names — same trick `13_build_sfincs.py` already uses for its own forwarding. File keys confirmed against the installed `hydromt_sfincs` source: weirs → `weirfile`, drainage_structures → `drnfile`, subgrid → `sbgfile` (verify `retreat`'s `write_man_tif=True` doesn't also touch `manningfile`).
- **Verify during implementation**: `mod.weirs.create(..., merge=True)`'s actual append-vs-replace behavior in the installed `hydromt_sfincs` version — this design assumes it appends to (not replaces) the skeleton's existing coastal-protection weir.
- Needs `adapted_root.mkdir(parents=True, exist_ok=True)` before opening the skeleton (mirrors `13_build_sfincs.py`'s own `sfincs_root.mkdir(...)`), and standard logging setup (`from src.log import setup_logging; log = setup_logging(snakemake.log[0])`) — **never** `import snakemake` explicitly (Snakemake injects that name into the script's namespace already; an explicit import shadows/breaks it).

## 4. New rule file: `workflow/rules/18b_adapt_post.smk` (1 rule) — revised 2026-08-13

```python
rule adapt_metrics_post:
    input:
        baseline_sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        baseline_flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        attribution_mask_tif   = results_path("{basin_id}/runs/{scenario}/sfincs/attribution_mask.tif"),  # NEW -- from rule attribution_mask, section 4a
        landuse        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        delta_polygon  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        measure_data   = lambda wildcards: strategy_measure_input_paths(wildcards.strategy),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
    params:
        strategy_def         = lambda wildcards: STRATEGY_DEFS[wildcards.strategy],
        measures_def         = MEASURES_DEFS,
        adaptation_root      = ADAPT_CATALOGUE_ROOT,
        # doubles as `scenario_root` for adaptation_method_post.dispatch_rules --
        # both the SfincsModel(water_level/output component) reads AND the
        # attribution_mask.tif lookup resolve against this same folder.
        baseline_sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        skeleton_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        hmin = config["metrics"]["hmin"], urban_code = config["metrics"]["urban_landuse_code"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
    log: "logs/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/18b_adapt_post.log"
    script: "../scripts/18b_adapt_post.py"
```

Note `baseline_flood_map_tif` is back as a real, directly-read input (unlike the original in-memory design) — the file-based measure chain needs an actual starting raster path for its first measure's own `flood_map_path`, it can't start from an in-memory `xr.DataArray`.

### `workflow/scripts/18b_adapt_post.py` (NEW) — pseudocode, revised 2026-08-13

1. `flood_map_path = str(input.baseline_flood_map_tif)` — the running path threaded through the measure chain, reassigned after each measure.
2. Loop over `params.strategy_def["measures"].items()`, in order:
   - `measure_def = params.measures_def[measure_type]`
   - resolve `locations`/`dep_subgrid` string params via `adaptation_input_path`
   - `result = adaptation_method_post.dispatch_rules(measure_type, flood_map_path=flood_map_path, scenario_root=params.baseline_sfincs_root, measure_def=measure_def, catalog_path=<TBD, see Open items>, output_dir=Path(output.metrics_csv).parent, method="postprocessing", **resolved_params)`
   - `flood_map_path = result["out_raster"]` — next measure reads THIS measure's own output.
3. After the loop, `flood_map_path` is (or gets copied/renamed to) `output.flood_map_tif`.
4. `metrics = compute_risk_metrics(<reload flood_map_path via rioxarray>, da_dep, input.landuse, input.delta_polygon, urban_code=params.urban_code)` — reused unmodified; needs `da_dep` (land-domain reference grid) from a `compute_max_inundation(params.baseline_sfincs_root, params.skeleton_root, input.sea_mask, ...)` call alongside, since only `da_hmax` ever gets persisted to a raster.
5. Write `output.metrics_csv` with `{"basin_id":..., "scenario":..., "strategy":..., "method": "post", **metrics}`.
6. Needs the same logging-setup / no-`import snakemake` conventions as `18a_adapt_pre.py` (see above).

## 4a. Attribution mask subsystem (NEW, 2026-08-13, upstream of `adapt_metrics_post`)

Classifies each pixel of a basin×scenario's flood map by dominant source (river / coastal / compound / baseline-"mean conditions") so `adaptation_method_post.py`'s measures can restrict an edit to the relevant class(es) (e.g. a coastal levee only ever removes classes {coastal, compound}, never river-sourced flooding). Runs **once per basin×scenario**, shared across every strategy — not once per strategy, and not duplicated per method (`pre` never touches it).

Ported from the user's own sibling `delta_model` project (`old_analyse.py`, called by `old_main.py`) into `workflow/src/attribution_plot.py`. Confirmed with the user: counterpart scenarios are matched RP-for-RP against `scenario`'s own `river_rp`/`surge_rp` (see `attribution_counterparts` in section 1), and the "baseline/mean conditions" class always compares against the fixed `default` scenario — **not** the low-RP `coast_only`/`river_only` scenarios (those instead serve as the RP=2 single-driver counterparts wherever a target scenario's own driver RP is 2, e.g. `coast_500`'s river-only counterpart).

**New rule file** `workflow/rules/18c_attribution_mask.smk`:

```python
rule attribution_mask:
    input:
        river_tif    = lambda wc: results_path(f"{wc.basin_id}/runs/{attribution_counterparts(wc.scenario)[0]}/visuals/max_flood_depth.tif"),
        coastal_tif  = lambda wc: results_path(f"{wc.basin_id}/runs/{attribution_counterparts(wc.scenario)[1]}/visuals/max_flood_depth.tif"),
        baseline_tif = results_path("{basin_id}/runs/default/visuals/max_flood_depth.tif"),
        skeleton_inp = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),   # mod_ref, for the diagnostic plot's own basemap
    output:
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/sfincs/attribution_mask.tif"),
        attribution_mask_png = results_path("{basin_id}/runs/{scenario}/sfincs/attribution_mask.png"),
    params:
        skeleton_root = lambda wc: results_path(f"{wc.basin_id}/sfincs_skeleton"),
        threshold = config["metrics"]["hmin"],   # TBD -- confirm reuse of hmin vs a dedicated attribution threshold (original script default: 0.05 m)
    log: "logs/{basin_id}/runs/{scenario}/18c_attribution_mask.log"
    script: "../scripts/18c_attribution_mask.py"
```

Output deliberately lands inside the scenario's own `sfincs/` folder (alongside `sfincs.inp`/`sfincs_map.nc`, written there by rules 13/16, reused unmodified) rather than under a separate `adaptation/` folder — this is what lets `adaptation_method_post.py`'s ported measures use a single `scenario_root` for both their `SfincsModel(root=scenario_root)` reads (water level / observation-point components) AND their `Path(scenario_root) / "attribution_mask.tif"` lookup, matching the other project's own convention. **Confirm with the user**: writing a non-SFINCS-native file into that folder doesn't break rules 13/16 (it doesn't — they only ever reference specific named files, never enumerate the directory), but it's a deliberate choice worth sign-off since that folder was previously "pure SFINCS working files."

**`workflow/scripts/18c_attribution_mask.py`** (NEW): thin wrapper — for this single basin×scenario, calls `src.attribution_plot.generate_attribution_maps(base_root=params.skeleton_root, attribution_runs=[(wildcards.scenario, input.river_tif, input.coastal_tif, input.baseline_tif, Path(output.attribution_mask_tif).parent)], threshold=params.threshold, ...)` — **not** the original script's hardcoded 4-scenario batch call.

**`workflow/src/attribution_plot.py`** needs real work before it's callable — currently:
- missing every import (`Path`, `rasterio`, `numpy`, `rioxarray`, `matplotlib`/`ListedColormap`/`BoundaryNorm`/`mpatches`, `SfincsModel`);
- ends with a top-level call to `generate_attribution_maps(root_folder=...)` that executes on import, referencing an undefined `root_folder` and the other project's own hardcoded scenario-folder names (`river_baseline`, `coast_baseline`, `river_flood`, `coastal_flood`, `compound`) — must become a plain importable function with that block deleted, driven entirely by this repo's actual `results_path`s and `attribution_counterparts`.

## 5. `workflow/src/adaptation_method_post.py` — revised 2026-08-13

Supersedes the original state-dict/bathtub-truncation design. Ported from the user's sibling `delta_model` project (reference copy kept at `workflow/src/old_postprocessing_adaptation.py`), **file-based** rather than in-memory:

- `dispatch_rules(measure_type, flood_map_path, scenario_root, measure_def, catalog_path, output_dir, landuse_path=None, method="postprocessing", **params) -> dict` — looks up `measure_type` in `selected_measures`, validates params via `validate_measure_params`, and calls the matching function with `flood_map_path`/`scenario_root`/`output_dir`/`landuse_path` plus the strategy's own resolved params. Returns `{"method": ..., "out_raster": str}` (`apply_retreat` also returns `"landuse_path"`, see below). `catalog_path` is accepted but never actually used/forwarded anywhere in the function body — confirmed dead, `18b_adapt_post.py` just passes `None`.
- `validate_measure_params` — currently **duplicated** from `adaptation_method_pre.py` rather than imported; consider re-deduplicating since the two are functionally identical (both only inspect `measure_def`/`params`, no model/state dependency).
- `selected_measures`: `offshore_barrier` (renamed from `offshore_island` mid-implementation, see Context — **resolved 2026-08-13**: `config/measures.yml`/`config/adaptation_strategies.yml`/both `selected_measures` dicts are now consistent. Along the way, `adaptation_method_pre.py`'s own rename had introduced a fresh bug — its function was defined as `apply_offshore_barriers` (plural) while `selected_measures` referenced `apply_offshore_barrier` (singular), a `NameError` on import — fixed by renaming the function to match the singular convention every other measure already uses), `coastal_and_river_levee`, `nbs_land_reclamation`, `water_retention` (both of these were explicitly deferred in the original plan for lacking `attribution_mask.tif` — now unblocked since that file gets built, section 4a), `coastal_levee`, `pumps`, `dike_ring`, `retreat`.
- Every measure except `retreat` reads `Path(scenario_root) / "attribution_mask.tif"` to restrict its edit to the relevant flood-source class(es) — e.g. `apply_coastal_levee` removes classes `{2 (coastal), 3 (compound)}` only if `elevation` exceeds the max coastal water level, read live from `SfincsModel(root=scenario_root, mode="r").get_component("water_level")` (**verify this hydromt_sfincs API** — same "confirm against the installed version" caveat class as `adaptation_method_pre.py`'s `weirs.create(merge=True)`). `apply_coastal_and_river_levee` additionally reads `.get_component("output").data["point_zs"]` for the river-side max water level.
- `apply_pumps`/`apply_water_retention` convert a discharge/storage-fraction into a uniform depth reduction over the relevant attribution classes (`volume / target_area`), clipped at 0 — same "graded, volume-based" approach as the pre-method's own docstrings describe, not a binary on/off.
- **`apply_retreat` — redesigned 2026-08-13**, no longer touches metrics files at all. Like every other measure it diverges from, it doesn't modify the hazard: the flood map is copied unchanged. But instead of rescaling `urban_exposed_km2` inside a `risk_metrics.csv` (the ported original's approach — required a file at `scenario_root/risk_metrics.csv` that this repo never produces, in a schema `flood_metrics.csv` doesn't match, and even if present would've been silently ignored since `18b_adapt_post.py` recomputes every metric fresh at the end of the chain regardless), it now **reclassifies landuse**: identifies eligible cells (`landuse == urban_code` AND `flood_depth > flood_threshold`), narrows to the deepest-flooded `retreat_fraction` of those via `np.nanquantile` (mirrors `adaptation_method_pre.py`'s own `apply_retreat` prioritization exactly), writes a new `retreat_landuse.tif` with those cells relabeled `target_code`, and returns it as `landuse_path`. `18b_adapt_post.py` threads `landuse_path` through the measure chain the same way it already threads `flood_map_path` (defaulting to `snakemake.input.landuse`, reassigned only when a measure's result carries a `"landuse_path"` key), and its own final `compute_risk_metrics()` call uses the threaded value — so retreat's effect flows through that single unified computation like every other measure's raster edit already does, no separate metrics file or schema reconciliation needed.

## 6. New rule file: `workflow/rules/19_adapt_compare.smk` (1 rule)

```python
rule adapt_compare:
    input:
        baseline_metrics = results_path("{basin_id}/runs/{scenario}/metrics/flood_metrics.csv"),
        pre_metrics  = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
        post_metrics = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
    output:
        comparison_csv = results_path("{basin_id}/runs/{scenario}/adaptation/compare/{strategy}/metrics_comparison.csv"),
    log: "logs/{basin_id}/runs/{scenario}/adaptation/compare/{strategy}/19_adapt_compare.log"
    script: "../scripts/19_adapt_compare.py"
```

`workflow/scripts/19_adapt_compare.py`: read all three CSVs, label each row's `method` (`"baseline"`/`"pre"`/`"post"`), add `strategy` column, concatenate into one long-format CSV, add delta columns (`pre - baseline`, `post - baseline`, `pre - post`) for `flooded_area_km2`, `urban_exposed_km2`, `volume_m3`.

## 7. `workflow/Snakefile` changes

Add after `include: "rules/17_flood_metrics.smk"`:
```python
include: "rules/18a_adapt_pre.smk"
include: "rules/18b_adapt_post.smk"
include: "rules/18c_attribution_mask.smk"   # NEW -- must precede 18b's own include if Snakemake ever enforces include order (it currently doesn't)
include: "rules/19_adapt_compare.smk"
```

Add after `_BUILD_OUTPUTS`, before `rule preprocess`:
```python
_ADAPT_OUTPUTS = []
if "pre" in METHODS:
    # Explicitly lists every output of adapt_run_event_pre/adapt_flood_metrics_pre
    # (not just one representative file per rule) -- Snakemake would build all of
    # a rule's declared outputs together regardless, but this matches
    # _BUILD_OUTPUTS_SCENARIO's own convention: every result file individually
    # named and directly targetable, not implicitly produced as a side effect.
    _ADAPT_OUTPUTS += (
        expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/01_inundation_ratio.png"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/02_flood_animation.mp4"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_timeseries.csv"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/max_flood_depth.tif"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
    )
if "post" in METHODS:
    _ADAPT_OUTPUTS += (
        expand(results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
        + expand(results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
               basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)
    )
if {"pre", "post"} <= set(METHODS):
    _ADAPT_OUTPUTS += expand(
        results_path("{basin_id}/runs/{scenario}/adaptation/compare/{strategy}/metrics_comparison.csv"),
        basin_id=BASINS, scenario=SCENARIOS, strategy=STRATEGIES)

rule adapt:
    """
    Run adaptation strategies against already-built baseline scenario runs
    (rule `build` must already have completed for the same basin/scenario).
    Opt-in: requires --config target_strategies=[...] (no default strategy
    set). target_methods defaults to both 'pre' and 'post'.

    Usage:  snakemake adapt --cores N --config target_strategies="['advance_1']"
    """
    input: _ADAPT_OUTPUTS
```

`attribution_mask.tif` is **not** added to `_ADAPT_OUTPUTS` directly — it's pulled in automatically as a dependency of `adapt_metrics_post` whenever `"post"` is in `METHODS`, same as every other intermediate file.

`rule all`'s `input:` (`_PREPROCESS_OUTPUTS + _BUILD_OUTPUTS`) is **not changed** — adaptation never runs as a side effect of `snakemake all`/`snakemake build`.

## Reuse matrix

| Existing file | Reused unmodified? | Note |
|---|---|---|
| `scripts/13_build_sfincs.py` | Yes | New rule repoints `skeleton_root`/`sfincs_root`. Cosmetic-only caveat: its own diagnostic plot title (derived from path segments) will show `"pre"`/`{strategy}` instead of basin/scenario — doesn't affect correctness. |
| `scripts/16_run_event.py` | Yes | Same repointing; same cosmetic plot-title caveat. |
| `scripts/17_flood_metrics.py` | Yes | Uses `snakemake.wildcards.basin_id`/`.scenario` directly — no mislabeling. CSV lacks a `strategy` column; `19_adapt_compare.py` adds it when concatenating. Also reused (unmodified) as the source of every scenario's own `runs/{scenario}/visuals/max_flood_depth.tif`, which `attribution_mask` (4a) reads for its river/coastal/baseline counterparts. |
| `src/adaptation_method_pre.py` | Yes, as-is | Called by new `18a_adapt_pre.py` exactly as its own module docstring already anticipates. **Except**: its `selected_measures` dict key is still `offshore_island`, not yet renamed to `offshore_barrier` to match `config/measures.yml` — see Open items. |
| `src/postprocessing.py` (`compute_max_inundation`, `compute_risk_metrics`) | Yes | `compute_risk_metrics` called directly by `18b_adapt_post.py` (on the final chained raster); `compute_max_inundation` still needed there too, for `da_dep`. |
| `src/sfincs_run.py` (`parse_sfincs_inp`, `forward_geometry_files`, `run_sfincs_subprocess`) | Yes | Used the same way existing scripts already use them. |

Genuinely new files: `rules/18a_adapt_pre.smk`, `rules/18b_adapt_post.smk`, `rules/18c_attribution_mask.smk`, `rules/19_adapt_compare.smk`, `scripts/18a_adapt_pre.py`, `scripts/18b_adapt_post.py`, `scripts/18c_attribution_mask.py`, `scripts/19_adapt_compare.py`, `src/adaptation_method_post.py`, `src/attribution_plot.py`.

## Open items to resolve during implementation (not blocking)

1. Whether `adaptation/dep_subgrid.tif` (referenced by the `retreat` strategy) is basin-specific or shared — confirm raw-data layout under `meta.root`.
2. Confirm `hydromt_sfincs` `mod.weirs.create(..., merge=True)` append-vs-replace behavior against the installed version.
3. `coastal_and_river_levee`/`pumps` post-method formulas remain approximations (per user decision, implemented but flagged) — needs review once real numbers come out, before treating them as physically meaningful.
4. ~~`adaptation_method_pre.py`'s `selected_measures` dict still keys off `offshore_island`~~ — **resolved 2026-08-13**: `config/measures.yml`, `config/adaptation_strategies.yml`, and both `adaptation_method_pre.py`/`adaptation_method_post.py`'s `selected_measures` dicts are now consistently `offshore_barrier`. Also fixed along the way: `adaptation_method_pre.py`'s own rename had left the actual function named `apply_offshore_barriers` (plural) while the dict referenced `apply_offshore_barrier` (singular) — a `NameError` on import, since `dispatch_rules` is imported at module load. Both modules now verified to import cleanly.
5. **(2026-08-13, still open)** `attribution_counterparts` (section 1) currently only resolves for scenarios with a same-RP single-driver sibling already defined in `scenarios.yml` — today that's `coast_500`/`river_500`/`compound_500` only (via `coast_only`/`coast_only_500`/`river_only`/`river_only_500`). Adapting `coast_100`/`coast_250`/`coast_1000`/`river_100`/`river_250`/`river_1000`/`compound_100` with the `post` method will raise until matching `*_only_<rp>` scenarios are added.
6. ~~`adaptation_method_post.py`'s `apply_retreat` expects a pre-existing `scenario_root/risk_metrics.csv`~~ — **resolved 2026-08-13**: redesigned to reclassify landuse instead of writing a metrics file at all; see section 5's updated `apply_retreat` bullet. No schema to reconcile anymore.
7. ~~`dispatch_rules`'s `catalog_path` parameter has no obvious source~~ — **resolved 2026-08-13**: confirmed by reading the function body that `catalog_path` is accepted but never used or forwarded anywhere internally (dead parameter, carried over from the ported code). `18b_adapt_post.py` passes `None`.
8. **(2026-08-13, still open)** Confirm `SfincsModel(...).get_component("water_level")` / `.get_component("output")` (used throughout `adaptation_method_post.py` for live water-level lookups) against the installed `hydromt_sfincs` version — same class of API-existence check as item 2 above, for a different component accessor.
9. **(2026-08-13, still open)** Placement of `attribution_mask.tif`/`.png` directly inside `runs/{scenario}/sfincs/` (alongside rules 13/16's own SFINCS working files, rather than a separate `adaptation/` folder) is a deliberate choice enabling a single `scenario_root` to serve both the `SfincsModel` reads and the attribution-mask lookup — confirm this is acceptable rather than an unwanted mixing of "baseline pipeline output" and "adaptation-only derived product."
10. **(2026-08-13, NEW)** `config/config.yml` was missing its `adaptation:` section (`strategies_file`/`measures_file`) entirely — `00_common.smk`'s `config["adaptation"]["strategies_file"]` lookup would have raised `KeyError` immediately. Now added and confirmed present.
11. **(2026-08-13, NEW, resolved)** `config/adaptation_strategies.yml` had two strategy names with dots (`advance_0.9`, `advance_0.4`) — invalid against `{strategy}`'s own `[A-Za-z0-9_-]+` wildcard regex (section 1), exactly the failure mode Verification step 4 was written to catch. Renamed to `advance_0_9`/`advance_0_4`.
12. **(2026-08-13, NEW, still blocking)** None of this is wired into the actual `Snakefile` yet: `18a_adapt_pre.smk`, `18b_adapt_post.smk`, and `19_adapt_compare.smk` all exist as files but aren't `include:`d, and `rule adapt`/`_ADAPT_OUTPUTS` (section 7) haven't been added — so no adaptation rule is reachable by Snakemake at all yet, not even for `-n`. `18c_attribution_mask.smk`/`.py` (section 4a) also don't exist yet, and `attribution_plot.py` is still exactly as broken as originally found (confirmed by re-attempting the import: `NameError: name 'Path' is not defined` — no imports, and the module-level `generate_attribution_maps(root_folder=...)` call block referencing an undefined `root_folder` is still there). These four pieces are what's left before `snakemake adapt` can run at all.

## Verification plan

1. **Non-regression dry run**: `snakemake all -n` — job list must be unchanged (new rules never appear, since `rule all` is untouched).
2. **Dry-run the new axis**, scoped to one basin/scenario/strategy/method:
   ```
   snakemake adapt -n --reason --config target_basins="[<basin_id>]" target_scenarios="['default']" target_strategies="['grey_protect_open']" target_methods="['pre']"
   ```
   Confirm the printed DAG chains `adapt_apply_pre → adapt_build_forcing_pre → adapt_run_event_pre → adapt_flood_metrics_pre`.
3. **Direct single-file targeting**:
   ```
   snakemake -n <results_dir>/<basin_id>/runs/default/adaptation/pre/grey_protect_open/flood_metrics.csv
   snakemake -n <results_dir>/<basin_id>/runs/default/adaptation/post/grey_protect_open/flood_metrics.csv
   ```
4. **Regex sanity check**: dry-run `target_strategies="['advance_0_9']"` and confirm a fabricated invalid name (e.g. `advance_0.9`, with a dot) is correctly rejected by `wildcard_constraints`.
5. **Attribution-mask dry run** (NEW): scoped to a scenario WITH counterparts defined (`coast_500`), confirm `attribution_mask` appears in the DAG ahead of `adapt_metrics_post`:
   ```
   snakemake -n --reason --config target_basins="[<basin_id>]" target_scenarios="['coast_500']" target_strategies="['grey_protect_open']" target_methods="['post']" adapt
   ```
   Then confirm a scenario WITHOUT counterparts defined (`coast_100`) fails fast with `attribution_counterparts`'s own clear error, not a downstream KeyError.
6. **Smoke test** (prerequisite: `rule build` already completed for the chosen basin/scenario). Start with `grey_protect_open` (single weir measure, no `retreat`, cheapest realistic strategy):
   ```
   snakemake --cores 4 --config target_basins="[<basin_id>]" target_scenarios="['default']" target_strategies="['grey_protect_open']" target_methods="['pre','post']" adapt
   ```
   Inspect both `flood_metrics.csv` files and `.../adaptation/compare/grey_protect_open/metrics_comparison.csv`.
7. **Retreat-specific smoke test** (exercises the scenario-dependent `flood_map_path` wiring and subgrid rebuild):
   ```
   snakemake --cores 4 --config target_basins="[<basin_id>]" target_scenarios="['default']" target_strategies="['retreat']" target_methods="['pre','post']" adapt
   ```
