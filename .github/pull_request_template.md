## What does this change

<!-- Which pipeline stage(s)/rule(s) does this touch, and why?
     e.g. "rule 09a (river depth): switch width-depth relation to X because Y" -->

## DAG / config impact

- [ ] New rule? Number chosen in topological order, not just appended
      (see CONTRIBUTING.md's "pipeline shape" section) — number: ____
- [ ] Touches `config/config.yml`, `config/data_catalogue.yml`, or
      `workflow/Snakefile`? Diff reviewed for accidental clobbering of
      unrelated settings (these are the highest-conflict shared files).
- [ ] New/renamed config keys are documented inline where they're defined.

## Checklist (see CONTRIBUTING.md for detail)

- [ ] `snakemake -n preprocess` / `snakemake -n build` (whichever applies)
      dry-runs cleanly against real data
- [ ] `python tests/check_code_health.py` passes (syntax + src-import
      resolution — catches broken imports that a dry-run misses)
- [ ] `pre-commit run --all-files` passes (ruff format/lint, pyright,
      pydocstyle)
- [ ] No unrelated changes bundled in
- [ ] If this changes model behavior (not just structure/docs): ran it
      end-to-end on at least one basin and sanity-checked the output
      (flooded area, inundation plots, etc. look physically reasonable)

## Notes for the reviewer

<!-- Anything non-obvious: a tradeoff you made, an assumption baked in,
     something you want a second opinion on. -->
