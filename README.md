# GCFM_UU

Snakemake pipeline that builds and runs SFINCS compound flood models (storm surge
plus river discharge) for river deltas from global datasets. For each delta it
derives the model domain, terrain, river channels, coastal protection and boundary
forcing, runs design flood scenarios, and tests adaptation strategies on them.

## What it produces

Everything is written under `<results_dir>/{basin_id}/`:

| path | content |
|---|---|
| `preprocessing_inputs/domain/` | delta polygon, model domain, elevation, land use, roughness, river network (raw, cleaned, depth-calibrated), coastal protection weir, SFINCS grid definition |
| `preprocessing_inputs/forcing/` | storm-tide hydrographs, surge and river forcing (`surge_forcing.nc`, `river_forcing.nc`) |
| `preprocessing_inputs/visuals/` | one diagnostic figure per preprocessing rule |
| `sfincs_skeleton/` | per-basin SFINCS model (`sfincs.inp`, `sfincs.weir`, `sfincs_subgrid.nc`) and `land_bed_level.nc`, the bed level every flood depth is computed on (open sea masked) |
| `spin_up/` | spin-up run, restart file and validation figures (when `sfincs.spinup.enabled`) |
| `runs/{scenario}/sfincs/` | event run (`sfincs_map.nc`) |
| `runs/{scenario}/visuals/` | maximum flood depth (`max_flood_depth.tif`), flood time series, animation |
| `runs/{scenario}/metrics/flood_metrics.csv` | flood-risk metrics (flooded shares of the land and urban area, depths) |
| `adaptation_lines/` | levee, mouth-closure and pump lines derived from the skeleton for `auto/` strategy locations, with a review figure |
| `runs/{scenario}/adaptation/{pre,post}/{strategy}/` | adapted runs and their flood metrics |
| `runs/metrics_comparison.csv` | all strategies and scenarios side by side |

## Quick start

1. Create the environment `<env>` (conda-forge; `hydromt-sfincs` is pinned and
   installed by pip). `environment.yml` names it `gcfm`;
   `environment.win-64.lock.yml` reproduces the exact Windows build. Details:
   [docs/install.md](docs/install.md).

   ```bash
   conda env create -f environment.yml
   conda activate gcfm
   ```

2. Get the input data into `<raw_data_root>`, laid out as the `file_path` entries
   of `config/data_catalogue.yml`. Sources, versions, licences and how to obtain
   each dataset: [docs/data.md](docs/data.md). You also need a SFINCS executable.

3. Copy the site settings and set the paths. There is no fallback: the
   pipeline stops and names the missing setting.

   ```bash
   cp config/site.example.yml config/site.yml
   ```

   | key in `config/site.yml` | points to |
   |---|---|
   | `results_dir` | `<results_dir>`, where outputs are written |
   | `raw_data_root` | `<raw_data_root>`, the root of every `file_path` in `config/data_catalogue.yml` |
   | `sfincs_exe` | `<sfincs_exe>`, the SFINCS binary (must be set for every run, dry runs included; any path works for `-n`) |
   | `adaptation_data_root` | `<adaptation_data_root>`, the per-basin adaptation measure files (only for the `adapt` target) |

   The environment variables `GCFM_RESULTS_DIR`, `GCFM_RAW_DATA_ROOT`,
   `GCFM_SFINCS_EXE` and `GCFM_ADAPTATION_DATA_ROOT` override the file when set, for example:

   ```bash
   export GCFM_RESULTS_DIR=/data/gcfm/results
   ```

   ```powershell
   $env:GCFM_RESULTS_DIR = "$HOME\gcfm\results"
   ```

   Because the variables win over the file, check for leftovers before the first
   run (`env | grep GCFM_` in bash, `Get-ChildItem env:GCFM_*` in PowerShell);
   how to unset them: [docs/install.md](docs/install.md#site-settings).

4. Check the job graph with a dry run on the smallest case (Ebro delta, basin
   `2433835`, scenario `default`); it lists 23 jobs:

   ```bash
   snakemake -n preprocess --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"
   ```

5. Run it. Without a target Snakemake runs `all` (preprocess + build).

   ```bash
   snakemake preprocess --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
   snakemake build      --cores 4 --config target_basins="[2433835]" target_scenarios="['default']"
   ```

### Targets

| target | runs | needs |
|---|---|---|
| `preprocess` | rules 01-11: domain, terrain, land use, river network, forcing, river depth and crest calibration | input data under `<raw_data_root>` |
| `build` | rules 13-17: SFINCS skeleton, spin-up, event runs, flood metrics | `preprocess` outputs, SFINCS |
| `all` | `preprocess` and `build` | |
| `adapt` | rules 13b, 18a-19: adaptation lines, strategies and comparison | `build` outputs, per-basin adaptation geometries, `--config target_strategies=...` |

Selecting basins, scenarios and strategies, adaptation runs and rule graphs:
[docs/running.md](docs/running.md).

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
| `config/scenarios.yml` | named flood scenarios: surge and river return periods (`surge_rp_yr`, `river_rp_yr`), sea-level rise |
| `config/data_catalogue.yml` | input datasets, paths relative to `<raw_data_root>` |
| `config/site.yml` | machine-specific paths (gitignored; template `config/site.example.yml`) |
| `config/measures.yml` | adaptation measures and their allowed parameters |
| `config/adaptation_strategies.yml` | strategies as combinations of measures, at the scales listed under `scales` |

Every key the code reads is required: there are no code defaults, and a missing
key fails at the rule that reads it. The `--config` keys above are the only
optional ones.

## Checks

```bash
pytest -q                              # synthetic tests in tests/unit
python tests/check_code_health.py      # syntax and src imports
pre-commit run --all-files             # ruff, pyright, pydocstyle, lock-file sync
```

`pytest -q` reports 325 passed in about 2 minutes. Tests that need real data
carry the `needs_data` marker and are excluded by default. `pre-commit` needs a
git clone and a one-time `pre-commit install`
([install.md](docs/install.md#pre-commit)).

## Documentation

| page | content |
|---|---|
| [docs/install.md](docs/install.md) | environment, SFINCS executable, site settings, verifying the install |
| [docs/data.md](docs/data.md) | every input dataset: version, source, licence, how to obtain it, where it goes (generated from `config/data_catalogue.yml`) |
| [docs/running.md](docs/running.md) | targets, `--config` keys, scenarios, adaptation runs, reruns, outputs |
| [docs/methods.md](docs/methods.md) | how the model is built: datums, grid, river depth and crest calibration, forcing, adaptation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | branching, pull requests, code and docstring style, pre-commit |
