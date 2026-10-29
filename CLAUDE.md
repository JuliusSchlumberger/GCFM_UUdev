# GCFM_UU

Snakemake pipeline that builds and runs SFINCS compound (storm surge + river) flood
models for river deltas from global datasets, and tests adaptation strategies on them.

## Environment

Conda env `gcfm`: portable `environment.yml`; exact Windows reproduction `environment.win-64.lock.yml` (pinned pip `hydromt-sfincs==2.0.0rc2`, runtime fix in `workflow/src/hydromt_sfincs_patch.py`).
`conda activate gcfm`, or `conda run -n gcfm <cmd>` when conda is not on PATH. Machine-specific notes: `CLAUDE.local.md`.
Site paths: `config/site.yml` (gitignored; copy `config/site.example.yml`), keys `results_dir`, `raw_data_root`, `sfincs_exe`, `adaptation_data_root` (adapt only); `GCFM_*` env vars override them.

## Entry points

- `snakemake preprocess --cores N`: rules 01-11, terrain, forcing, depth and crest calibration
- `snakemake build --cores N`: SFINCS build, spin-up, event run, flood metrics
- `snakemake all --cores N`: preprocess + build; the default target (`default_target`), so a bare `snakemake` runs it; any target takes `--config target_basins="[ID]" target_scenarios="['name']"`
- `snakemake adapt --cores N --config target_strategies="['advance_1']"`: needs `build` done; strategies with `auto/<name>` locations also run rule 12c `build_adaptation_lines` (writes `<results_dir>/{basin_id}/adaptation_lines/`)
- `python tools/<script>.py`: one-off data preparation
- `python analysis/<script>.py [basin_id]`: stand-alone figures and checks on finished results, including the diagnostics that are not rules (river-network map, upstream-boundary check, bifurcation options, GRDC and GESLA screens); shared helpers in `analysis/_common.py`
- `python plot_dag.py`: rule graph to `figs/dag/`

## Protected (change only with explicit go-ahead)

- `raw_data_root` (read only) and `results_dir` (pipeline outputs), as set in `config/site.yml`
- `config/config.yml`, `config/scenarios.yml`, `config/data_catalogue.yml`, `config/adaptation_strategies*.yml`, `config/measures.yml`
- `environment.yml`, `environment.win-64.lock.yml`, `.pre-commit-config.yaml` (pinned versions); `config/site.yml` (machine paths)
- Local, gitignored: `config/site.yml`, `CLAUDE.local.md`

## Changelog

Every change to code, config or results goes into `CHANGELOG.md` in the same session:
- **Placement:** newest first, under a dated heading `# YYYY-MM-DD: <topic> (- <initials>)`.
- **Author initials:** the person making the change. Take them from `CLAUDE.local.md` (`changelog initials`), or else from the initials of `git config user.name`. Ask if neither gives them.
- **Content:** what changed and why, with numbers where results moved.
- **Reverts and revisions:** when a change is reverted or revised later in the same session, edit its entry so it describes the final state. Don't add a correction entry on top.
- **Scope:** `CHANGELOG.md` is the one place for history; code and docs stay history-free.

## Smallest end-to-end case

Basin `2433835` (Ebro delta), scenario `default`; adaptation: strategy `advance_1`.
`advance_1` needs the per-basin adaptation geometries under `adaptation_data_root` (`{basin_id}/`, e.g. `2433835/offshore_island.geojson`).

## Baselines (`.baseline/`, gitignored; recorded at the commit in `.baseline/git_head.txt`)

| check | command | expected |
|---|---|---|
| lint | `ruff check . --statistics --no-fix` (ruff 0.13.0, pre-commit's copy) | 575 (F821: 540) |
| unit tests | `pytest -q` | 336 passed (`needs_data` excluded; 0 marked) |
| code health | `python tests/check_code_health.py` | OK, 82 files |
| quick-look figures | `pytest -q tests/unit/test_quicklook.py` | 16 PNGs in `.baseline/figures/` (synthetic delta) |
| preprocess DAG | `snakemake -n preprocess --forceall --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"` | 15 jobs |
| build DAG | same, target `build` | 22 jobs |

CI does not use `environment.yml`: `.github/workflows/ci.yml` pip-installs its own list for `tests/unit`.
When a test, or a `src` module a test imports, gains a module-level third-party import, add it there.
