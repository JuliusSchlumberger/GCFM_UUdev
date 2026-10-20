# Working rules

Standing rules for every project. Project-specific facts (entry points, protected
paths, run/test commands, baseline results) go in the repo's CLAUDE.md; machine
paths go in CLAUDE.local.md (gitignored).

## 1. Token efficiency

- Locate before reading: grep, glob, AST, `git grep` first; read a file in full
  only when the match needs context. Read line ranges, not whole files.
- Never read the same file twice in a session; never re-read after an edit.
- Independent tool calls go in one batch. Independent sub-tasks go to parallel
  subagents that return a short structured summary, never file contents.
- Do not echo file contents, diffs or command output back in chat; refer to
  file:line. Do not restate the request or summarize what was just done unless
  asked.
- Verification proportional to the change: lint only for text edits; full
  tests for logic; expensive checks (dry runs, parity runs) once, at the end.
- Stop when the task is done. No "next steps" lists, no offers.
- Long sessions: compact after each completed task; keep a running one-line
  task list rather than re-deriving state.

## 2. How to work

- State the mode at the start of each task:
  - **develop**: results are meant to change. Name the intended effect first
    (which outputs, in which direction, roughly how much), then build, then show
    the effect happened.
  - **maintain**: results must not change. Refactor, document, clean; prove
    parity.
  Unclear which: ask, one line.
- Small task (< ~5 files, familiar code): just do it. Larger or unfamiliar:
  map first (what calls what, what is affected), propose in a few lines, then
  build.
- One concern per run. Finish it, verify, stop. Do not start the next task
  unasked.
- Report in chat. No new .md files unless asked. Deferred work goes to TODO.md,
  one line each with file:line.
- Ask before anything irreversible: deleting outputs, renaming output files or
  CLI flags, changing defaults, touching protected files.

## 3. Scope

- Fix what was asked. Other problems found on the way: TODO.md, then move on.
- A change that alters results outside the stated intended effect is a bug,
  not a bonus; log it and stop.
- Stale code is deleted (git keeps it), never moved to an archive folder.
- Protected files (named per repo): output patterns, config defaults, data
  catalogs, pinned environments. Change only with explicit go-ahead.

## 4. Verification

- Before declaring a task done: lint and the fast test suite. State what was run
  and the result; never say "verified" without naming the check.
- Match the proof to the change:
  - docs/comments only: lint
  - moves/splits: moved defs AST-identical, defined once
  - refactor (maintain): parity run of old vs new on the same inputs; compare
    output hashes, DataFrame equality, raster statistics
  - development: before/after on a small fixed test case, showing the intended
    effect in numbers and, for spatial or time-series outputs, a quick-look
    figure (see section 8)
  - config: materialize and compare parsed values
  - workflow rules: dry-run job count against the recorded baseline
- Baselines (job counts, output hashes, reference figures) live in a gitignored
  scratch dir and are recorded in the repo CLAUDE.md with the command that
  reproduces them.

## 5. Documentation standard

Documentation is for future readers of the code, not a record of the session.

- No history anywhere in code or docs: no "previously", "updated to", "fixed",
  "refactored", phases, dates. Git is the changelog.
- Docstrings: one imperative summary line, max ~80 chars. More lines only for
  what the signature cannot say: units, datums, sign conventions, assumptions,
  valid ranges, the equation or paper implemented, non-obvious side effects or
  failure modes.
- No Parameters/Returns sections that repeat type hints. Document a parameter
  only when name and type leave it unclear.
- No docstrings on private helpers, trivial wrappers, dunders or tests unless a
  non-obvious assumption is involved.
- Comments explain why, never what. Delete comments that paraphrase code. Keep
  comments that encode a physical or numerical assumption or a reference.
- Module docstring: one or two lines, purpose and entry point. No file
  listings, author or date headers.
- README is the front door and a map, ~150 lines: what it does, what it
  produces, quick start with exact commands, layout, configuration, links.
  Depth lives in docs/ pages (install, data, running, hpc, methods).
- If a name needs a comment to explain it, rename instead.
- Unsure whether to document: leave it out.

## 6. Code conventions

- Config over constants: every path, threshold, scenario name, resource limit
  comes from one config file; one source of truth, derived values computed.
- No absolute paths in code. Data root, scratch and site settings from config
  or environment variables.
- Fail loudly. No bare `except`; catch specific errors; re-raise with context.
  Retries only around errors known to be transient. Drivers exit non-zero on
  any failed step.
- Validate inputs at the boundary (shape, CRS, units, nodata) and assume them
  valid inside.
- Units and conventions in names: `depth_m`, `elevation_msl_m`, `rp_years`,
  `slr_scenario`. One name per concept across the whole repo.
- Importing a module has no side effects; `--help` never runs anything.
  Scripts use argparse and a `main()`; library code is pure, I/O at the edges.
- Deterministic by default: explicit seeds, sorted iteration over files and
  dict keys, fixed dtypes and nodata values.
- Logging, not print, in library code; print only in CLI output meant for the
  user.
- Type hints on public signatures; ruff clean; no unused parameters.
- Prefer boring code: standard library and the project's existing dependencies
  before a new package.
- New functionality comes with its smoke test and, if it writes spatial or
  time-series output, its quick-look figure in the same change.

## 7. Portability

- Site-specific values (cluster, partitions, accounts, data roots, interpreter
  paths) live in one committed `*.example.*` file with comments, and a
  gitignored real copy. Code fails with a clear message if the real copy is
  missing.
- The institution or cluster is named once, in the example file, and nowhere
  else. Docs say "the HPC", "<data_root>", "<env>" with a fenced example after
  the placeholder.
- Required data: table generated from the data catalog (dataset, version,
  source, licence, how to register). Never assume data on a shared drive is
  available.
- LICENSE and CITATION.cff exist before anything is shared.

## 8. Tests and quick-look outputs

- Every core module has at least a smoke test that imports it and runs the main
  function on synthetic inputs small enough to run in seconds.
- Tests that need external data carry a marker and are excluded from the
  default run.
- Test names state the check (`test_rp100_exceeds_rp10`, not `test_eva`).
- Numerical checks assert properties (monotonic in return period, mass
  conserved, nodata preserved, bounds), not exact floats, unless the value is a
  documented constant.
- Quick-look outputs: every pipeline stage that writes a raster, grid, vector
  layer or time series has a small plotting helper that renders one panel per
  key variable to a scratch dir. After a development change, render before and
  after side by side, look at the image, and state what changed. These figures
  are diagnostics, not deliverables: fixed colour scales, titles with the
  stage, scenario and git hash, no styling work.
- Reference figures for the standard test case live with the baselines; a
  maintain change must reproduce them.

## 9. Data safety

- Never write to, move or delete anything under a protected data root; reading
  is fine. Test outputs go to a gitignored scratch directory.
- Never run a full-grid workflow or a long job to "check" something; use the
  smallest scenario or a dry run.
- Progress monitors and anything that touches a running batch are run by the
  user.

## 10. Report format

Findings, one row each:

    file:line | issue | evidence | proposed fix | impact (none/text/behaviour) | status

Evidence is concrete: "0 references (grep + AST)", "doc says --rp, parser has
--return-period". No finding, no row. Needs-decision items first, as a checklist.

## 11. Bringing an existing repo up to these rules

One phase per session, in this order, each verified:

1. Tooling and repo CLAUDE.md (entry points, commands, protected paths,
   baselines).
2. Read-only audit: map, dependency graph, file classification, plan.
3. Smoke tests and quick-look helpers for core modules.
4. Delete stale and duplicate files.
5. Documentation trim to section 5; report preserved domain facts.
6. Refactors from the audit, one module per session.
7. Consistency sweep (dead functions, config keys, docs vs code, naming, error
   handling), report first, fix on approval.
8. Portability pass and README as front door; cold-start subagent follows the
   README and reports gaps.

New code follows sections 1 to 10 from the first commit.

# This repo

## Smallest end-to-end scenario

Basin `2433835` (Ebro delta), scenario `default`. Use it for every check that
needs the pipeline; never run other basins or the full basin list to "check".

    snakemake build --cores N --config target_basins="[2433835]" target_scenarios="['default']"

## Baselines

Run from the repo root in the `hmt_sfincs_dev` env. Dry-run counts use
`--forceall` so they do not depend on what is already on disk.

| check | command | expected |
|---|---|---|
| unit tests | `pytest -q` | 45 passed |
| code health | `python tests/check_code_health.py` | OK, 77 files, no errors |
| lint | `pre-commit run --all-files` | all hooks pass |
| preprocess DAG | `snakemake -n preprocess --forceall --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"` | 22 jobs |
| build DAG | `snakemake -n build --forceall --cores 1 --config target_basins="[2433835]" target_scenarios="['default']"` | 25 jobs |

Update a row only when a change is meant to alter it, and say so in the commit.
