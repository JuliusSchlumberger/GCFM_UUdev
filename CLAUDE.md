# GCFM_UU

Snakemake pipeline that builds and runs SFINCS compound (storm surge + river) flood
models for river deltas from global datasets, and tests adaptation strategies on them.

## Environment

Conda env `hmt_sfincs_dev` (`environment.yml`), plus an editable `hydromt_sfincs` (see README).
`conda activate hmt_sfincs_dev`, or `conda run -n hmt_sfincs_dev <cmd>` when conda is not on PATH.
Required env vars: `GCFM_RESULTS_DIR`, `GCFM_RAW_DATA_ROOT`, `GCFM_SFINCS_EXE`.

## Entry points

- `snakemake preprocess --cores N`: rules 01-12, terrain, forcing, depth and crest calibration
- `snakemake build --cores N`: SFINCS build, spin-up, event run, flood metrics
- `snakemake all --cores N`: preprocess + build (default); any target takes `--config target_basins="[ID]" target_scenarios="['name']"`
- `snakemake adapt --cores N --config target_strategies="['advance_1']"`: needs `build` done
- `python tools/<script>.py`: one-off data preparation; `python plot_dag.py`: rule graph

## Protected (change only with explicit go-ahead)

- `$GCFM_RAW_DATA_ROOT` (read only) and `$GCFM_RESULTS_DIR` (pipeline outputs)
- `config/config.yml`, `config/scenarios.yml`, `config/data_catalogue.yml`, `config/adaptation_strategies*.yml`, `config/measures.yml`
- `environment.yml`, `.pre-commit-config.yaml` (pinned versions)
- Other protected paths: <unknown: ask>

## Smallest end-to-end case

Basin `2433835` (Ebro delta), scenario `default`; adaptation: <unknown: ask>

## Baselines (`.baseline/`, gitignored; recorded at the commit in `.baseline/git_head.txt`)

| check | command | expected |
|---|---|---|
| lint | `ruff check . --statistics --no-fix` (ruff 0.13.0, pre-commit's copy) | 710 (F821: 576) |
| unit tests | `pytest -q` | 213 passed (`needs_data` excluded; 0 marked) |
| code health | `python tests/check_code_health.py` | OK, 76 files |
| quick-look figures | `pytest -q tests/unit/test_quicklook.py` | 19 PNGs in `.baseline/figures/` (synthetic delta) |
| preprocess DAG | `snakemake -n preprocess --forceall --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"` | 22 jobs |
| build DAG | same, target `build` | 25 jobs |

CI does not use `environment.yml`: `.github/workflows/ci.yml` pip-installs its own list for `tests/unit`.
When a test, or a `src` module a test imports, gains a module-level third-party import, add it there.
