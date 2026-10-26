# Running the pipeline

All commands run from the repository root inside `<env>` (see
[install.md](install.md)), with `config/site.yml` in place. Outputs go to
`<results_dir>/{basin_id}/`, for example:

```
/data/gcfm/results/2433835/
```

## Targets

Without a target Snakemake runs `all`.

| target | rules | runs | needs |
|---|---|---|---|
| `preprocess` | 01-11 | domain, land use, terrain, protection levels, river network, forcing, model grid, river depth and crest calibration, diagnostic checks | input data under `<raw_data_root>`, SFINCS (rule 10 runs it) |
| `build` | 13-17 | SFINCS skeleton, spin-up, per-scenario forcing, event runs, flood metrics | `preprocess` outputs, SFINCS |
| `all` | 01-17 | `preprocess` and `build` | as above |
| `adapt` | 13b, 18a-19 | adaptation lines, adaptation strategies and their comparison | `build` outputs, adaptation geometries, `target_strategies` |

```bash
snakemake preprocess --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
snakemake build      --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
```

Preview any run with `-n` (dry run). For basin `2433835` and scenario
`default`, on an empty `<results_dir>`, `preprocess` has 23 jobs, `build` 27 and
`all` 30. Threads per rule are set under `resources.threads` in
`config/config.yml`: the SFINCS rules take the whole `--cores` budget as solver
threads, so a SFINCS run occupies the machine while it runs.

Rules that read large raw inputs (`get_elevation`, `get_river_network`,
`get_boundary_forcings`) claim one unit of the `raw_io`
resource. The workflow profile `workflow/profiles/default/config.yaml` limits it
to 1, so these reads do not compete for one disk; on a parallel file system
raise it, for example `--resources raw_io=4`.

Useful Snakemake flags: `--forceall` (rerun every job of the target),
`--forcerun <rule>` (rerun a rule and everything downstream), `--until <rule>`,
`--keep-going` (continue other basins after a failure), `--unlock` (after a
killed run).

## `--config` keys

These keys are set on the command line only, never in `config/config.yml`. Values
are parsed as YAML.

| key | default | example | effect |
|---|---|---|---|
| `target_basins` | every `BasinID2` in the delta-polygon dataset | `target_basins="[2433835]"` | basins to run; an unknown id stops the run before the DAG is built |
| `target_scenarios` | `['default']` | `target_scenarios="['default','coast_100_03SLR']"` | scenarios from `config/scenarios.yml`; an unknown name stops the run |
| `target_strategies` | `[]` (no adaptation runs) | `target_strategies="['advance_1','retreat']"` | strategy names or families from `config/adaptation_strategies.yml`; a family selects all its scales |
| `target_method` | `['pre','post']` | `target_method="['post']"` | adaptation methods to run |

## Scenarios

`config/scenarios.yml` defines named scenarios; the name is the `{scenario}`
wildcard. A `default` entry is required.

| key | required | values |
|---|---|---|
| `surge_rp_yr` | yes | COAST-RP return period (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000), `Tide` (average tide, needs `boundary_forcings.surge.hydrograph.enabled`), or null (no surge forcing) |
| `river_rp_yr` | yes | return period in [2, 1000] years, `Mean` (mean discharge), or null (no discharge) |
| `slr_m` | yes | global-mean sea-level rise (m), scaled per station by its AR6 fingerprint; 0 = none |
| `event` | adaptation runs only | `coastal`, `river` or `compound`: picks the per-event parameter values of a strategy; every scenario in `config/scenarios.yml` sets it |

The forcing mode follows from the two return periods: both set (including `Tide`
or `Mean`) is `compound`, one set is `coastal_only` or `river_only`, both null is
an error. Return periods and `slr_m` are applied when the
scenario is built (rule 13), so adding or changing a scenario reruns only that
scenario's build, event run and metrics, never preprocessing or the spin-up.

## Spin-up

`sfincs.spinup.enabled` (default `true`) runs one basin-level spin-up (rule 14)
of `spinup_days` at constant mean discharge and calm sea; every scenario's event
run starts from its restart file at `tref + spinup_days`. With
`enabled: false`, rules 14 and 15 are skipped and each event run starts at
`tref` from the skeleton's initial water level (sea cells at the calm-sea level,
land dry). The `adapt` target requires `enabled: true`: its rules read the
spin-up restart and the spin-up map.

## Adaptation runs

`adapt` needs, for every selected basin and scenario:

- `sfincs.spinup.enabled: true` (with `target_strategies` set and the spin-up
  off, the workflow stops at DAG construction) and the `build` outputs,
  including the spin-up restart and spin-up map;
- for strategies whose measures have `locations:`, the geometry files under
  `<adaptation_data_root>` listed in [data.md](data.md#adaptation-geometries),
  drawn per basin: `{basin_id}` in a path is the basin id (for example `advance`
  on basin 2433835 needs `2433835/offshore_island.geojson`), and a
  missing file stops the workflow at DAG construction;
  `accommodate` and `retreat` need none;
- nothing extra for `locations: auto/<name>` (`grey_protect_open`,
  `protect_closed`): rule `build_adaptation_lines` (13b) derives these lines from
  the basin's skeleton into `<results_dir>/{basin_id}/adaptation_lines/`, and
  reruns with the skeleton. Review `adaptation_lines.png` there. The rule fails if
  a mouth closure does not seal its river; `--keep-incomplete` keeps the figure
  for inspection;
- an `event` key on each scenario;
- for the attribution mask (rule 18c, used by `post` and by water-retention
  measures): a river-only sibling scenario with the same `river_rp_yr` and
  `surge_rp_yr: null`, and a coastal-only sibling with the same `surge_rp_yr`
  and `river_rp_yr: null`.

```bash
snakemake adapt --cores 4 --config target_basins="[2433835]" target_scenarios="['compound_100']" target_strategies="['advance_1']"
```

`advance_1`, the smallest adaptation case, reads
`<adaptation_data_root>/2433835/offshore_island.geojson`, and `compound_100`
has the siblings `river_only_100` and `coast_only_100`. On an empty `<results_dir>`
the dry run (`-n`) of this command lists 39 jobs, including `preprocess` and the
`build` of all three scenarios.

| method | rule | what it does |
|---|---|---|
| `pre` | 18a | edits a copy of the skeleton (weirs, drainage structures, subgrid) and reruns SFINCS for the scenario |
| `post` | 18b | applies the measures to the baseline `max_flood_depth.tif` without SFINCS (cheap, approximate) |

With both methods, rule 19 writes `runs/metrics_comparison.csv` from every
`flood_metrics.csv` under the basin's `runs/`. Methods and strategies:
[methods.md](methods.md#adaptation).

## Reruns

Snakemake reruns a job when an input is newer than its outputs, or when the
rule's code, parameters or input set changed since the last run. Parameters are
scoped so that a scenario change reruns only scenario-level rules. To ignore
code edits (for example a comment change in a script):

```bash
snakemake build --cores 4 --rerun-triggers mtime params input --config target_basins="[2433835]"
```

`snakemake --cleanup-metadata <file>` forgets the provenance of one output
without touching the file.

When a job fails, Snakemake deletes all its declared outputs, including SFINCS
results the solver already finished writing (for example `sfincs_map.nc` or the
restart file). The failed rule must be rerun; upstream outputs are kept.

## Outputs

The output layout under `<results_dir>/{basin_id}/` is listed in the
[README](../README.md#what-it-produces). Each preprocessing rule writes a
diagnostic figure to `preprocessing_inputs/visuals/`; the depth and crest
calibration keeps its disposable model and per-round state in
`preprocessing_inputs/depth_crest_calibration/` and its figures in
`preprocessing_inputs/visuals/10_calibration/`.

## Rule graphs

`plot_dag.py` renders the rule graph and job DAG of each target with Graphviz
`dot` to `figs/dag/`:

```bash
python plot_dag.py --targets preprocess build
python plot_dag.py --targets preprocess --format png --no-dag
```

Options: `--targets` (default `preprocess build`), `--out-dir` (default
`figs/dag`), `--format` (`svg`, `png` or `pdf`), `--no-dag` (rule graph only);
`python plot_dag.py --help` lists them. The script passes no `--config`, so the
graphs cover the defaults: every basin in the delta-polygon dataset and the
`default` scenario. The rule graph does not depend on the basin count; the job
DAG has one node per job and grows with it (use `--no-dag` for a quick look).
For one basin, call Snakemake directly:

```bash
snakemake preprocess --dag --forceall --config target_basins="[2433835]" | dot -Tsvg -o figs/dag/preprocess_2433835.svg
```

## Analysis scripts

`analysis/` holds stand-alone figures and checks that read finished results.
They have no command-line options: set the basin and scenarios in the constants
at the top of each script, then run it, for example
`python analysis/flood_maps.py`. They read `config/site.yml` through
`analysis/_common.py` (`site_path`) and save their figures under
`<results_dir>/{basin_id}/runs/`; Snakemake does not track them.
