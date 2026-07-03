# Contributing to GCFM_UU

This document describes how we work together on this repository. It exists so
that reviews focus on logic and results, not on re-explaining conventions
every time. If something here is unclear or gets in the way, raise it — this
is a living document for a two-person project, not a fixed policy.

## The pipeline shape matters for how we branch

Unlike a typical app with independent modules, `workflow/` is one Snakemake
DAG made of **numbered, ordered pipeline stages** (`workflow/rules/00_*.smk`
through `16_*.smk`, each with a matching `workflow/scripts/NN_*.py`), sharing
one config schema (`config/config.yml`) and one shared library
(`workflow/src/*.py`). There usually isn't a clean "your folder vs. my
folder" split — plan branches around that:

- **Adding a new pipeline stage** (a new rule number): claim the number
  first (message the other person / open a draft PR early) so we don't both
  end up writing "rule 17". Insert it in true topological order, not just
  appended at the end — see the rule docstrings for how existing rules
  reason about DAG position (e.g. why rule 10 can't run earlier than it
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
  `main` directly — all changes land via pull request (PR). The repo is now
  public, so GitHub branch protection rules are available for free — this
  needs to be turned on explicitly (it isn't automatic just from making the
  repo public); until it is, nothing actually blocks a direct push to `main`,
  so treat the PR-only rule as a personal discipline commitment for now. To
  enable real enforcement: repo **Settings → Branches → Add branch
  protection rule**, branch name pattern `main`, enable "Require a pull
  request before merging" and "Require status checks to pass before merging"
  (select the CI job(s) from `.github/workflows/ci.yml`). Once set up, a
  green CI run + checked-off PR checklist becomes an actual merge
  requirement, not just a convention.
- Create one branch per pipeline stage or task, off the latest `main`:
  `feature/<short-description>` (e.g. `feature/discharge-eva`) for new rules
  or capabilities, `fix/<short-description>` for bug fixes.
- Keep branches short-lived (days, not weeks). Open a PR early — even as a
  draft — if you want feedback before the work is finished. A long-lived
  branch against a DAG this interconnected accumulates conflicts fast.
- Update your branch from `main` regularly while you're the only one working
  on it:

  ```bash
  git fetch origin
  git rebase origin/main
  ```

  Resolve conflicts locally, in small increments, rather than letting them
  pile up. Once a PR is open and someone else may have looked at or pulled
  your branch, switch to `git merge origin/main` instead of rebasing, so we
  don't rewrite shared history.

## Config changes

`config/config.yml` is organised by *who consumes a setting*, not by feature
area — e.g. `datum_correction` and `protection_levels` are top-level because
more than one rule reads them, `boundary_setup` is separate from
`boundary_forcings` because one produces forcing data and the other
configures how the built model consumes it. Before adding a new key:

- Check whether it belongs under an existing section (which rule(s) will
  read it?) rather than defaulting to a new top-level block.
- Add an inline comment explaining *why* the value is what it is or *why*
  it's grouped where it is — matching the existing comment style in the
  file — not just what it does.
- If you rename or move a key, `grep` for the old path across `workflow/`,
  `tests/`, and this repo's Claude Code memory file (see below) — config
  key renames are the easiest thing to half-finish, since a stale reference
  won't fail until that specific code path runs.

## Before opening a PR

- [ ] `snakemake -n preprocess` and `snakemake -n build` (or the relevant
      target) complete without errors — this catches DAG/config errors but
      **not** broken imports (see below).
- [ ] If you renamed or deleted anything in `workflow/src/`, verify every
      `from src.X import Y` across `workflow/scripts/`, `workflow/src/`, and
      `tests/` still resolves. `ast.parse`/a dry-run will NOT catch this —
      a broken import only fails the first time that specific script
      actually executes, which can be well after it's merged. A quick way to
      check: try importing every touched module directly with `workflow/` on
      `sys.path`.
- [ ] Pre-commit hooks pass (`pre-commit run --all-files` — see below).
      This covers formatting/lint (ruff) and docstring style (pydocstyle,
      Google style) automatically.
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
- **Docstrings**: Google style (`Args:` / `Returns:`), checked by
  `pydocstyle` where present — see `pyproject.toml`'s `add_ignore` list.
  Docstrings are **not mandatory on every function** (missing-docstring
  rules D102/D103 are deliberately ignored); the convention actually in use:
  - `workflow/src/*.py` (shared library functions): thorough Google-style
    docstrings, since these are called across multiple rules.
  - `workflow/rules/*.smk`: the rule's own triple-quoted docstring explains
    *why* it sits where it does in the DAG, not what the script does
    line-by-line — see almost any existing rule for the pattern.
  - `workflow/scripts/*.py` (pipeline-step scripts): a short module
    docstring is enough; `workflow/scripts/` is excluded from the
    pydocstyle pre-commit hook entirely. Inline comments only where the
    *why* isn't obvious from the code (matching the project's general
    "don't restate what the code does" convention).
- Naming: `snake_case` for functions/variables, `PascalCase` for classes,
  descriptive names over abbreviations except where a term is standard in
  the field (e.g. `dem`, `eva`, `gpd`, `amax`, `rp`).

## Environment management (Windows / conda)

- The environment is defined in `environment.yml` (conda), env name
  `hmt_sfincs_dev`. Create it fresh with:

  ```bash
  mamba env create -f environment.yml
  ```

  or update an existing one with `mamba env update -f environment.yml --prune`.
- **`hydromt-sfincs` needs a separate editable install from source** — the
  version this project depends on isn't just `pip install`-able from
  `environment.yml` alone. After creating the environment, clone and install
  it per [README.md](README.md):

  ```bash
  git clone https://github.com/Deltares/hydromt_sfincs.git
  pip install -e hydromt_sfincs
  ```

  (see also Deltares's own [dev install guide](https://deltares.github.io/hydromt_sfincs/latest/dev_guide/dev_install.html#dev-env)
  if you need to set up a from-scratch dev environment rather than adding
  the editable install to `hmt_sfincs_dev`.)
- If your work needs a new package, install it into `hmt_sfincs_dev`
  locally, then `git add environment.yml` (even without editing it) and
  commit. The `sync-environment-yml` pre-commit hook
  (`tools/sync_environment.py`) automatically regenerates the file from the
  live environment, strips conda's machine-specific trailing `prefix:` line
  (always your local absolute install path — must never be committed, since
  `conda env create` would try to use it as the literal install target on
  the next machine unless overridden with `-n`/`-p`), and compares it to
  what's staged. If it's stale, the hook rewrites `environment.yml` and
  fails the commit (same as the `ruff --fix` hook) — review the diff,
  `git add environment.yml` again, and re-commit.
  - This hook only runs when `environment.yml` is already part of the
    commit — it doesn't fire on unrelated commits, so drift on your machine
    won't pollute commits that don't touch the environment.
  - It needs `conda` resolvable via `$CONDA_EXE` (set automatically when a
    conda environment is active) or on `PATH`.
  - To run it manually without committing:
    `python tools/sync_environment.py`.
- Don't pin exact build strings unless there's a known compatibility issue —
  prefer version-only pins so the environment stays resolvable on a
  different machine/OS patch level.
- The SFINCS executable itself (`sfincs.simulation.sfincs_exe` in
  `config.yml`) is a separately licensed/downloaded binary, not part of the
  conda environment or this repo. Rules that only *assemble* SFINCS input
  files (`build_sfincs`) don't need it; rules that actually *run* the
  solver (`run_spinup`, `run_event`) do — point that config value at your
  own local copy.

## Setting up pre-commit (one-time, after cloning)

`.pre-commit-config.yaml` being in the repo does **nothing by itself** —
it only takes effect once you've run, from inside `hmt_sfincs_dev`:

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
`sync environment.yml with hmt_sfincs_dev`, etc.) as `Passed`/`Failed`, not
silently do nothing. You can also make a throwaway commit and check that the
hooks' output appears before it completes.

**If hooks behave differently between a manual `pre-commit run` and an
actual `git commit`** (e.g. some hooks silently don't run, or one fails with
a "Python was not found" / exit-code-9009-style error): check
`.git/hooks/pre-commit`'s `INSTALL_PYTHON` line — it hardcodes the exact
interpreter path pre-commit was installed from at `pre-commit install` time.
If it points at the wrong/an old conda env (this has happened once already
on this repo — it was pointing at a stale, differently-named env), rerun
`pre-commit install` from inside `hmt_sfincs_dev` to refresh it.

## Reference memory (optional, if you use Claude Code)

`Reference_memory.txt` at the repo root is a living reference an AI
assistant maintains across sessions on this project — current pipeline
structure, config schema, known issues, rationale for non-obvious design
choices. It's not a substitute for code comments and isn't required reading
to contribute, but if you're using Claude Code on this repo, keeping it in
sync with structural changes (new rules, renamed config keys, etc.) means
the next session — yours or the other person's — doesn't have to
re-discover context that already got worked out once.

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
