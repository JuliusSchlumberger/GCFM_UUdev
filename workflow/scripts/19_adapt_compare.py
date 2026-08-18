"""
19_adapt_compare.py — Aggregate a basin's baseline/pre/post flood_metrics.csv
across every scenario x strategy currently in play (SCENARIOS x STRATEGIES)
into one long-format comparison CSV.
"""

from pathlib import Path

import pandas as pd

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

def _read_labeled(paths: list[str], method: str, strategy_from_path: bool) -> pd.DataFrame:
    """Read each CSV in `paths`, tagging every row with `method` and a
    `strategy` (the containing folder name -- e.g. .../pre/{strategy}/
    flood_metrics.csv -- since only the post-method's own CSV already
    carries a strategy column, and even that one is overwritten here for
    consistency). Baseline runs carry no strategy at all."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["method"] = method
        df["strategy"] = Path(p).parent.name if strategy_from_path else "no strategy"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

baseline_df = _read_labeled(snakemake.input.baseline_metrics, "baseline", strategy_from_path=False)
pre_df = _read_labeled(snakemake.input.pre_metrics, "pre", strategy_from_path=True)
post_df = _read_labeled(snakemake.input.post_metrics, "post", strategy_from_path=True)

long_df = pd.concat([baseline_df, pre_df, post_df], ignore_index=True)

long_df.to_csv(snakemake.output.comparison_csv, index=False)
log.info(
    f"Wrote comparison CSV: {snakemake.output.comparison_csv} "
    f"({len(long_df)} rows, {baseline_df['scenario'].nunique()} scenario(s), "
    f"{pre_df['strategy'].nunique()} strategy/strategies)"
)
