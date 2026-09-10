"""
19_adapt_compare.py — Aggregate a basin's baseline/pre/post flood_metrics.csv
across EVERY scenario x strategy ever run for this basin (not just whichever
target_strategies/target_scenarios were passed to this specific `snakemake
adapt` invocation) into one long-format comparison CSV.

Strategies are normally authored/tested a handful at a time, not all in one
invocation, so building the table purely from snakemake.input (this run's
own declared dependencies) would only ever reflect whichever strategies this
particular invocation happened to target, dropping every other
already-computed strategy's rows. Instead, every flood_metrics.csv actually
sitting on disk under this basin's own runs/ folder is discovered directly
via glob, independent of this invocation's own target_strategies/
target_scenarios -- snakemake.input is still declared on the rule
(19_adapt_compare.smk) so Snakemake knows to RERUN this rule whenever the
currently-targeted strategy/scenario's own metrics change, but the script
itself always recompiles the full table from whatever is actually on disk,
self-healing if a strategy folder is later deleted/renamed rather than
accumulating stale rows forever.
"""

from pathlib import Path

import pandas as pd

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

# comparison_csv is results_path("{basin_id}/runs/metrics_comparison.csv") --
# its own parent IS this basin's runs/ folder, the root every scenario/
# strategy/method's own flood_metrics.csv lives under.
runs_root = Path(snakemake.output.comparison_csv).parent


def _read_labeled(paths: list[Path], method: str, strategy_from_path: bool) -> pd.DataFrame:
    """Read each CSV in `paths`, tagging every row with `method` and a
    `strategy` (the containing folder name -- e.g. .../pre/{strategy}/
    flood_metrics.csv -- since only the post-method's own CSV already
    carries a strategy column, and even that one is overwritten here for
    consistency). Baseline runs carry no strategy at all."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["method"] = method
        df["strategy"] = p.parent.name if strategy_from_path else "no strategy"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


baseline_paths = sorted(runs_root.glob("*/metrics/flood_metrics.csv"))
pre_paths = sorted(runs_root.glob("*/adaptation/pre/*/flood_metrics.csv"))
post_paths = sorted(runs_root.glob("*/adaptation/post/*/flood_metrics.csv"))

baseline_df = _read_labeled(baseline_paths, "baseline", strategy_from_path=False)
pre_df = _read_labeled(pre_paths, "pre", strategy_from_path=True)
post_df = _read_labeled(post_paths, "post", strategy_from_path=True)

long_df = pd.concat([baseline_df, pre_df, post_df], ignore_index=True)

long_df.to_csv(snakemake.output.comparison_csv, index=False)
log.info(
    f"Wrote comparison CSV: {snakemake.output.comparison_csv} "
    f"({len(long_df)} rows from {len(baseline_paths)} baseline, {len(pre_paths)} pre, "
    f"{len(post_paths)} post flood_metrics.csv file(s) discovered under {runs_root})"
)
