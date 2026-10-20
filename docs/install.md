# Install

Four things are needed: the conda environment, the input data
([data.md](data.md)), a SFINCS executable and the site settings.

## Conda environment

`environment.yml` lists the direct dependencies with major.minor pins
(conda-forge, Python 3.11, Snakemake 9.21, HydroMT 1.3). It creates the
environment `<env>`, named in its `name:` line, for example:

```
gcfm
```

```bash
conda env create -f environment.yml
conda activate gcfm
```

Creating the environment takes about 12 minutes (conda solve, then the pip
phase). Update an existing environment with
`conda env update -f environment.yml --prune`. `mamba` works in place of
`conda`.

To create the environment at a path `<prefix>` instead of by name, use `-p`:

```bash
conda env create -p /opt/envs/gcfm -f environment.yml
```

When `conda` is not on `PATH`, call it by its full path (for example
`~/miniforge3/bin/conda` or `%USERPROFILE%\miniforge3\Scripts\conda.exe`), and
prefix pipeline commands with `conda run -n gcfm` or `conda run -p <prefix>`:

```bash
conda run -p /opt/envs/gcfm snakemake -n preprocess --cores 1
```

| file | use |
|---|---|
| `environment.yml` | portable environment, any OS |
| `environment.win-64.lock.yml` | full `conda env export` of the reference Windows environment (exact builds); reproduces it on Windows only |

```bash
conda env create -f environment.win-64.lock.yml
```

`tools/sync_environment.py` regenerates the lock file from the live
environment; the `sync-environment-lock` pre-commit hook runs it when the lock
file is staged (see [CONTRIBUTING.md](../CONTRIBUTING.md#environment-management-conda)).

### hydromt-sfincs

`hydromt-sfincs` is pinned to `2.0.0rc2` and installed by pip from the `pip:`
section of both files; no source clone is needed. The same section installs
`pyshtools` (geoid synthesis), `cdsapi` (download tools) and `line-profiler`.

`workflow/src/hydromt_sfincs_patch.py` fixes a memory bug in that release at
run time: `SfincsSubgridTable.write_netcdf` copies float32 subgrid arrays into
float64 buffers, which runs out of memory on about 1e6 active cells. The scripts
that write a SFINCS model (rules 10, 13 and 18a) call
`use_float32_subgrid_write()` first. The fix applies only to `2.0.0rc2`; any
other installed version logs a warning and runs unpatched.

## SFINCS executable

The pipeline calls a SFINCS binary; it is not part of the conda environment or
this repository and is licensed and obtained separately.

| | |
|---|---|
| needed by | rules that run the solver: `modelled_depth_estimation` (10), `run_spinup` (14), `run_event` (16), `adapt_run_event_pre` (18a) |
| not needed by | every other rule, including `build_sfincs`, which only writes input files |
| version | SFINCS v2.3.0 ("Faber" release), multithreaded build (`_mt`); the restart files follow the `sfincs.YYYYMMDD.HHMMSS.rst` naming |
| source | Deltares SFINCS releases, https://github.com/Deltares/SFINCS/releases |
| licence | see the SFINCS repository |
| other versions | <to be documented> |
| threads | each solver rule takes the whole `--cores` budget and sets `OMP_NUM_THREADS` for the SFINCS process |

Point `sfincs_exe` in the site settings at the binary.

## Site settings

Three machine-specific paths live in `config/site.yml` (gitignored). Create it
from the template and fill in the keys:

```bash
cp config/site.example.yml config/site.yml
```

| key | environment variable | points to | needed by |
|---|---|---|---|
| `results_dir` | `GCFM_RESULTS_DIR` | `<results_dir>`, where outputs are written | every run |
| `raw_data_root` | `GCFM_RAW_DATA_ROOT` | `<raw_data_root>`, root of every `file_path` in `config/data_catalogue.yml` and every measure file in `config/adaptation_strategies.yml` | every run |
| `sfincs_exe` | `GCFM_SFINCS_EXE` | `<sfincs_exe>`, the SFINCS binary | solver rules only |

For example:

```yaml
results_dir: /data/gcfm/results
raw_data_root: /data/gcfm/raw_data
sfincs_exe: /opt/sfincs/bin/sfincs
```

Quote Windows paths with single quotes or use forward slashes. All code reads
the keys through `src.io.site_setting`; `analysis/` scripts use
`_common.site_path`. There is no fallback: a missing key stops the run with a
message naming the key and its variable.

An environment variable, when set and non-empty, overrides its key:

```bash
export GCFM_RAW_DATA_ROOT=/data/gcfm/raw_data
```

```powershell
$env:GCFM_RAW_DATA_ROOT = "D:\gcfm\raw_data"
```

Because the `GCFM_*` variables win over the file, check for leftovers from an
earlier setup before the first run, and unset any you do not intend:

```bash
env | grep GCFM_
unset GCFM_RESULTS_DIR GCFM_RAW_DATA_ROOT GCFM_SFINCS_EXE
```

```powershell
Get-ChildItem env:GCFM_*
Remove-Item env:GCFM_RESULTS_DIR                                           # this session
[Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", $null, "User")   # for good
```

## Verifying the install

From the repository root, inside `<env>`:

```bash
pytest -q                            # synthetic unit and smoke tests, no data needed
python tests/check_code_health.py    # syntax and src-import check
pre-commit run --all-files           # ruff, pyright, pydocstyle, lock-file sync
snakemake -n preprocess --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"
```

`pre-commit` needs the one-time setup under [Pre-commit](#pre-commit). The tests
and the health check need no input data. Quick-look figure titles carry the git
commit hash; outside a git clone (for example an unpacked ZIP) they show `nogit`
and the tests still pass. The dry run needs the site
settings and the delta polygons under `<raw_data_root>`; for basin `2433835`
(Ebro) and scenario `default` it lists 22 jobs for `preprocess` on an empty
`<results_dir>` (25 for `build`). Tests that need real data carry the
`needs_data` marker and are excluded by default.

## Pre-commit

Contributors install the git hooks once per clone (ruff, pyright, pydocstyle,
lock-file sync):

```bash
pre-commit install
```

Details: [CONTRIBUTING.md](../CONTRIBUTING.md#setting-up-pre-commit-one-time-after-cloning).
