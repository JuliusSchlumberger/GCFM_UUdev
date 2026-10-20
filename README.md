# GCFM_UU

Snakemake pipeline that builds and runs SFINCS compound flood models (storm surge
plus river discharge) for river deltas from global datasets. For each delta it
derives the model domain, terrain, river channels, coastal protection and boundary
forcing, runs design flood scenarios, and tests adaptation strategies on them.

## What it produces

Everything is written under `$GCFM_RESULTS_DIR/{basin_id}/`:

| path | content |
|---|---|
| `preprocessing_inputs/domain/` | delta polygon, model domain, elevation, land use, roughness, river network (raw, cleaned, depth-calibrated), coastal protection weir, SFINCS grid definition |
| `preprocessing_inputs/forcing/` | storm-tide hydrographs, surge and river forcing (`surge_forcing.nc`, `river_forcing.nc`) |
| `preprocessing_inputs/visuals/` | one diagnostic figure per preprocessing rule |
| `sfincs_skeleton/` | per-basin SFINCS model (`sfincs.inp`, `sfincs.weir`, `sfincs_subgrid.nc`) |
| `spin_up/` | spin-up run, restart file and validation figures (when `sfincs.spinup.enabled`) |
| `runs/{scenario}/sfincs/` | event run (`sfincs_map.nc`) |
| `runs/{scenario}/visuals/` | maximum flood depth (`max_flood_depth.tif`), flood time series, animation |
| `runs/{scenario}/metrics/flood_metrics.csv` | flood-risk metrics (flooded shares of the land and urban area, depths) |
| `runs/{scenario}/adaptation/{pre,post}/{strategy}/` | adapted runs and their flood metrics |
| `runs/metrics_comparison.csv` | all strategies and scenarios side by side |

## Quick start

1. Create the environment (conda-forge; `hydromt-sfincs` is pinned and installed by pip):

   ```bash
   mamba env create -f environment.yml
   conda activate hmt_sfincs_dev
   ```

2. Set three environment variables. There is no fallback: the pipeline stops and
   names the missing variable.

   | variable | points to |
   |---|---|
   | `GCFM_RESULTS_DIR` | `<results_dir>`, where outputs are written |
   | `GCFM_RAW_DATA_ROOT` | `<data_root>`, the root of every `file_path` in `config/data_catalogue.yml` |
   | `GCFM_SFINCS_EXE` | `<sfincs_exe>`, the SFINCS binary (needed only by rules that run the solver) |

   For example, in PowerShell for the current session:

   ```powershell
   $env:GCFM_RESULTS_DIR = "D:\gcfm\results"
   $env:GCFM_RAW_DATA_ROOT = "D:\gcfm\raw_data"
   $env:GCFM_SFINCS_EXE = "C:\sfincs\sfincs.exe"
   ```

3. Check the job graph with a dry run on the smallest case (Ebro delta, basin
   `2433835`, scenario `default`):

   ```bash
   snakemake -n preprocess --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"
   ```

4. Run it. Always name the target: without one Snakemake runs the first rule in
   `workflow/Snakefile`, which is `adapt`.

   ```bash
   snakemake preprocess --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
   snakemake build      --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
   ```

### Targets

| target | runs | needs |
|---|---|---|
| `preprocess` | rules 01-11: domain, terrain, land use, river network, forcing, river depth and crest calibration | data catalogue under `$GCFM_RAW_DATA_ROOT` |
| `build` | rules 13-17: SFINCS skeleton, spin-up, event runs, flood metrics | `preprocess` outputs |
| `all` | `preprocess` and `build` | |
| `adapt` | rules 18a-19: adaptation strategies and comparison | `build` outputs, per-basin adaptation geometries under `$GCFM_RAW_DATA_ROOT` |

Command-line keys (`--config`), all optional except `target_strategies` for `adapt`:

| key | default | example |
|---|---|---|
| `target_basins` | every basin found | `target_basins="[2433835]"` |
| `target_scenarios` | `['default']` | `target_scenarios="['coast_100','river_100']"` |
| `target_strategies` | none | `target_strategies="['advance_1']"` |
| `target_method` | `['pre','post']` | `target_method="['pre']"` |

Adaptation example:

```bash
snakemake adapt --cores 4 --config target_basins="[2433835]" target_scenarios="['default']" target_strategies="['advance_1']"
```

Render the rule graphs to `figs/dag/`:

```bash
python plot_dag.py --targets preprocess build
```

## Layout

```
workflow/Snakefile    entry point: includes the rules, defines the targets
workflow/rules/       one .smk file per numbered stage (00_common.smk: shared config parsing)
workflow/scripts/     one script per rule, same number
workflow/src/         shared library imported by the scripts
config/               pipeline configuration (see below)
tools/                one-off data preparation scripts (downloads, dataset fixes)
analysis/             stand-alone figures and checks on finished results
tests/unit/           synthetic unit and smoke tests (pytest)
tests/check_code_health.py   syntax and `src` import check over the repo
plot_dag.py           rule-graph rendering
```

## Configuration

| file | holds |
|---|---|
| `config/config.yml` | all pipeline settings, grouped by the rules that read them |
| `config/scenarios.yml` | named flood scenarios: surge and river return periods (`surge_rp_yr`, `river_rp_yr`), discharge multiplier, sea-level rise |
| `config/data_catalogue.yml` | input datasets, paths relative to `$GCFM_RAW_DATA_ROOT` |
| `config/measures.yml` | adaptation measures and their allowed parameters |
| `config/adaptation_strategies.yml` | strategies as combinations of measures, at the scales listed under `scales` |

Every key the code reads is required: there are no code defaults, and a missing
key fails at the rule that reads it. The `--config` keys above are the only
optional ones.

## Checks

```bash
pytest -q                              # synthetic tests in tests/unit
python tests/check_code_health.py      # syntax and src imports
pre-commit run --all-files             # ruff, pyright, pydocstyle, environment.yml sync
```

Tests that need real data carry the `needs_data` marker and are excluded by default.

## Further reading

- [CONTRIBUTING.md](CONTRIBUTING.md): branching, pull requests, code and docstring
  style, environment and pre-commit setup.
