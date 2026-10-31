# Contributing to GCFM_UU

This document describes how we work together on this repository. It exists so
that reviews focus on logic and results, not on re-explaining conventions
every time. If something here is unclear or gets in the way, raise it — this
is a living document for a two-person project, not a fixed policy.

## The pipeline shape matters for how we branch

Unlike a typical app with independent modules, `workflow/` is one Snakemake
DAG made of **numbered, ordered pipeline stages** (`workflow/rules/00_*.smk`
through `19_*.smk`, each except `00_common.smk` with matching
`workflow/scripts/NN*_*.py`), sharing
one config schema (`config/config.yml`) and one shared library
(`workflow/src/*.py`). There usually isn't a clean "your folder vs. my
folder" split — plan branches around that:

- **Adding a new pipeline stage** (a new rule number): claim the number
  first (message the other person / open a draft PR early) so we don't both
  end up writing "rule 20". Insert it in true topological order, not just
  appended at the end — see the rule docstrings for how existing rules
  reason about DAG position (e.g. why rule 11 can't run earlier than it
  does despite operating on elevation).
- **Improving an existing rule/script/src function**: normal feature branch,
  scoped to the files that rule actually touches.
- **Shared, high-conflict files** — `config/config.yml`, `workflow/Snakefile`,
  `config/data_catalogue.yml` — get touched by almost every change. Keep
  edits to these small and rebase often; if you're adding a new config
  section, check `config.yml`'s existing structure first (see "Config
  changes" below) so two people don't independently invent two different
  places for conceptually similar settings.

## Branching model

- `main` is always the current, working state of the model. Nobody pushes to
  `main` directly — all changes land via pull request (PR). GitHub enforces
  this only with a branch protection rule (repo **Settings → Branches → Add
  branch protection rule**, pattern `main`, "Require a pull request before
  merging" and "Require status checks to pass before merging" with the CI
  job from `.github/workflows/ci.yml`); without one it is a team rule.
- Create one branch per pipeline stage or task, off the latest `main`:
  `feature/<short-description>` (e.g. `feature/discharge-eva`) for new rules
  or capabilities, `fix/<short-description>` for bug fixes.
- Keep branches short-lived (days, not weeks). Open a PR early — even as a
  draft — if you want feedback before the work is finished. A long-lived
  branch against a DAG this interconnected accumulates conflicts fast.
- Update your branch from `main` regularly — see the exact commands in
  "Step-by-step: branch, work, sync, push, PR" below. Rebase while you're
  still the only one working on the branch; switch to merging once it's
  pushed and someone else may have pulled it, so shared history doesn't get
  rewritten.

## Step-by-step: branch, work, sync, push, PR

Concrete commands for the whole cycle, from creating your branch through
opening the PR. Substitute your branch name for `feature/my-change`.

**1. Create your branch, off the latest `main`:**

```bash
git checkout main
git pull origin main
git checkout -b feature/my-change   # or: git checkout feature/my-change, if it already exists
```

**2. Work on it** — edit files, then stage and commit as usual:

```bash
git add <files>
git commit -m "Short imperative summary of this change"
```

Commit as often as makes sense to you locally — there's no requirement to
squash before pushing (see "Pull requests" below for what happens at merge
time).

**3. Pull updates from `main`** while you're the only one working on your
branch (i.e. before you've pushed it, or before anyone else has pulled it):

```bash
git fetch origin
git rebase origin/main
```

If `main` moved since you branched, this replays your commits on top of the
latest `main` — resolve any conflicts it reports, `git add` the resolved
files, then `git rebase --continue`. Once your branch is pushed and the
other person may have looked at or pulled it, switch to `git merge
origin/main` instead of rebasing (rebasing rewrites commit hashes, which
breaks anyone else's copy of the branch).

**4. Push your branch to GitHub:**

```bash
git push -u origin feature/my-change
```

The `-u` is only needed the first time you push this branch — it links your
local branch to `origin/feature/my-change` so plain `git push`/`git pull`
work afterward without naming the branch each time.

**5. Open the pull request.** After `git push`, the terminal prints a direct
link, e.g.:

```
remote: Create a pull request for 'feature/my-change' on GitHub by visiting:
remote:      https://github.com/<owner>/<repo>/pull/new/feature/my-change
```

Open that link (or go to the repo on GitHub — it shows a "Compare & pull
request" banner for any recently-pushed branch), set the base branch to
`main`, and fill in the PR template. If you have the `gh` CLI installed, `gh
pr create` does the same thing from the terminal.

Repeat steps 2-4 for every further round of changes on the same branch — the
PR updates automatically with each push, no need to reopen it.

## Config changes

`config/config.yml` is organised by *who consumes a setting*, not by feature
area — e.g. `datum_correction` and `flopros_range` are top-level because
more than one rule reads them, `boundary_setup` is separate from
`boundary_forcings` because one produces forcing data and the other
configures how the built model consumes it. Before adding a new key:

- Check whether it belongs under an existing section (which rule(s) will
  read it?) rather than defaulting to a new top-level block.
- Add an inline comment explaining *why* the value is what it is or *why*
  it's grouped where it is — matching the existing comment style in the
  file — not just what it does.
- Every key is required: the code has no defaults, so a new key must be set
  in the yml before the code reads it.
- If you rename or move a key, `grep` for the old path across `workflow/`,
  `tools/`, `analysis/`, `tests/` and `docs/` — config
  key renames are the easiest thing to half-finish, since a stale reference
  won't fail until that specific code path runs.

## Before opening a PR

- [ ] `snakemake -n preprocess` and `snakemake -n build` (or the relevant
      target) complete without errors — this catches DAG/config errors but
      **not** broken imports (see below).
- [ ] If you renamed or deleted anything in `workflow/src/`, verify every
      `from src.X import Y` across `workflow/scripts/`, `workflow/src/`,
      `tools/`, `analysis/` and `tests/` still resolves. `ast.parse`/a
      dry-run will NOT catch this — a broken import only fails the first time that specific script
      actually executes, which can be well after it's merged. A quick way to
      check: try importing every touched module directly with `workflow/` on
      `sys.path`.
- [ ] `python tests/check_code_health.py` passes (syntax + `src` import
      resolution; CI does not run it).
- [ ] `pytest` passes (synthetic unit tests in `tests/unit`; CI runs them).
- [ ] Pre-commit hooks pass (`pre-commit run --all-files` — see below).
      This covers formatting/lint (ruff), type checks (pyright) and
      docstring style (pydocstyle) automatically.
- [ ] New config entries are documented inline in `config.yml` (see "Config
      changes" above), and new data sources are documented in
      `config/data_catalogue.yml` following the existing entry format.
- [ ] No unrelated changes bundled into this PR.

## Code style

- **Formatting & linting**: [`ruff`](https://docs.astral.sh/ruff/) (format +
  check), enforced via pre-commit — not black/flake8.
- **Type checking**: [`pyright`](https://github.com/microsoft/pyright), via
  pre-commit. Currently informational (`typeCheckingMode = "off"` in
  `pyproject.toml`) — flags obvious issues without blocking on strict typing.
- **Docstrings**: one imperative summary line (max ~80 characters). Add
  lines only for what the signature cannot say: units, datums, sign
  conventions, assumptions, valid ranges, the method or paper implemented,
  side effects. No `Args:`/`Returns:` sections that repeat the type hints;
  when a section is needed, use Google style (`pydocstyle` runs with the
  Google convention; `pyproject.toml`'s `add_ignore` skips missing-docstring
  and missing-argument checks, so neither is required). No docstrings on
  private helpers, trivial wrappers or tests unless an assumption is
  non-obvious.
  - `workflow/src/*.py` (shared library): public functions get the summary
    line plus the non-obvious facts above.
  - `workflow/rules/*.smk`: the rule's own triple-quoted docstring explains
    *why* it sits where it does in the DAG, not what the script does
    line-by-line — see almost any existing rule for the pattern.
  - `workflow/scripts/*.py` (pipeline-step scripts): a short module
    docstring is enough; `workflow/scripts/` is excluded from every
    pre-commit hook (top-level `exclude` in `.pre-commit-config.yaml`),
    so ruff and pydocstyle do not check it there. Inline comments only where the
    *why* isn't obvious from the code (matching the project's general
    "don't restate what the code does" convention).
- Naming: `snake_case` for functions/variables, `PascalCase` for classes,
  descriptive names over abbreviations except where a term is standard in
  the field (e.g. `dem`, `eva`, `gpd`, `amax`, `rp`).

## Environment management (conda)

- The portable environment is `environment.yml` (direct dependencies,
  major.minor pins). It names the environment `<env>`, for example:

  ```bash
  mamba env create -f environment.yml   # creates env "gcfm"
  conda activate gcfm
  ```

  or update an existing one with `mamba env update -f environment.yml --prune`.
- `environment.win-64.lock.yml` is the full `conda env export` of the
  reference Windows environment (exact builds); create from it to reproduce
  that machine exactly on Windows.
- `hydromt-sfincs` is pinned to `2.0.0rc2` in the `pip:` section of both
  files, so the commands above install it; no source clone is needed.
  `workflow/src/hydromt_sfincs_patch.py` holds a runtime fix for that release
  (float32 subgrid write buffers), called by the scripts that write a SFINCS
  model (rules 11, 13a, 17b).
- If your work needs a new package, add it with a major.minor pin to
  `environment.yml`, install it into `<env>`, then `git add
  environment.win-64.lock.yml` (even without editing it) and commit. The
  `sync-environment-lock` pre-commit hook (`tools/sync_environment.py`)
  regenerates the lock file from the live environment named in its `name:`
  line, strips conda's machine-specific trailing `prefix:` line (your local
  absolute install path, which `conda env create` would otherwise use as the
  install target on the next machine), and compares it to what's staged. If
  it's stale, the hook rewrites the lock file and fails the commit (same as
  the `ruff --fix` hook): review the diff, `git add
  environment.win-64.lock.yml` again, and re-commit.
  - This hook only runs when the lock file is already part of the commit, so
    drift on your machine won't pollute commits that don't touch the
    environment.
  - It needs `conda` resolvable via `$CONDA_EXE` (set automatically when a
    conda environment is active), on `PATH`, or in a standard
    miniforge3/miniconda3/anaconda3 install under your home directory.
  - To run it manually without committing:
    `python tools/sync_environment.py`.
- Don't pin exact build strings in `environment.yml`: version-only pins keep
  it solvable on other machines and operating systems.
- The SFINCS executable itself is a separately licensed/downloaded binary,
  not part of the conda environment or this repo. Rules that only
  *assemble* SFINCS input files (`build_sfincs`) don't need it; rules that
  actually *run* the solver (`run_spinup`, `run_event`, the depth
  calibration, pre-adaptation runs) do: point `sfincs_exe` at your own local
  copy (see "Site settings" below).

## Site settings (one-time, after cloning)

Three settings are specific to your own machine:

- the results directory: where pipeline outputs get written.
- the raw-data root: where raw input data lives locally (every
  `file_path` in `data_catalogue.yml` and every measure file in
  `adaptation_strategies.yml` is relative to it).
- the SFINCS executable: your local SFINCS binary.

They live in `config/site.yml`, which is gitignored, so a `git pull` can never
reset or conflict with your local paths. Create it from the committed
template and fill in the three keys:

```bash
cp config/site.example.yml config/site.yml
```

| Key | Override variable | Sets | Required for |
|---|---|---|---|
| `results_dir` | `GCFM_RESULTS_DIR` | results directory (`config["results_dir"]`) | every Snakemake run |
| `raw_data_root` | `GCFM_RAW_DATA_ROOT` | raw-data root (`CATALOGUE["meta"]["root"]`) | every Snakemake run |
| `sfincs_exe` | `GCFM_SFINCS_EXE` | SFINCS executable | only rules that run the solver |

All code reads them through `src.io.site_setting` (`workflow/src/io.py`);
`analysis/` scripts use `_common.site_path`. There is no fallback: the
pipeline stops with a message naming the missing key and its variable.

An environment variable, when set and non-empty, overrides its key (useful
for CI or batch jobs):

```bash
export GCFM_RESULTS_DIR=/data/gcfm/results
```

```powershell
$env:GCFM_RESULTS_DIR = "$HOME\gcfm\results"
```

Verify they're picked up with a dry run; the printed input/output paths
should reflect your own directories:

```bash
snakemake preprocess -n --cores 1
```

## Setting up pre-commit (one-time, after cloning)

`.pre-commit-config.yaml` being in the repo does **nothing by itself**;
it only takes effect once you've run, from inside `<env>`:

```bash
pre-commit install
```

This writes the actual git hook to `.git/hooks/pre-commit` (a
per-clone file, not tracked by git — every fresh clone needs this run once).
Without it, `git commit` won't trigger any checks at all; they'll only run
if you invoke `pre-commit run` manually.

To confirm it's active:

```bash
pre-commit run --all-files
```

should list every hook (`ruff format`, `ruff check`, `pyright`, `pydocstyle`,
`sync environment.win-64.lock.yml with the project conda env`, etc.) as
`Passed`/`Failed`, not silently do nothing. You can also make a throwaway
commit and check that the hooks' output appears before it completes.

**If hooks behave differently between a manual `pre-commit run` and an
actual `git commit`** (e.g. some hooks silently don't run, or one fails with
a "Python was not found" / exit-code-9009-style error): check
`.git/hooks/pre-commit`'s `INSTALL_PYTHON` line — it hardcodes the exact
interpreter path pre-commit was installed from at `pre-commit install` time.
If it points at the wrong or an old conda env, rerun `pre-commit install`
from inside `<env>` to refresh it.

## Documentation

`docs/` holds the install, data, running and methods pages linked from the
README. When a change alters a rule, an output, a config key or a modelling
choice, update the page that describes it in the same pull request.
`docs/data.md` is generated: edit `config/data_catalogue.yml` and run
`python tools/make_data_table.py`.

## Commit messages

Short, imperative summary line (`Add EVA dual-estimator for AMAX series`),
with a blank line and more detail below if needed. Doesn't need to follow a
strict format (e.g. Conventional Commits) — clarity over ceremony.

## Pull requests

- Use the PR template (auto-filled when you open a PR).
- Request review from the other person once CI is green.
- The reviewer checks: does this integrate cleanly with the rest of the DAG
  (right topological position, no silently-stale references to anything
  renamed/moved), does it follow the conventions above, and do the results
  make physical/hydrological sense (not just "does it run").
- Squash-merge by default, to keep `main` history clean and one commit per
  logical change. Use a regular merge only if the individual commits are
  independently meaningful and you want to preserve them.

## Questions / disagreements about conventions

If a convention here doesn't fit a specific case, say so in the PR — we'd
rather adjust this document than force an awkward workaround.
