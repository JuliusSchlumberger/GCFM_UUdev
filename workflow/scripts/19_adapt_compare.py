"""Aggregate every baseline/pre/post flood_metrics.csv of a basin into one long CSV.

Run by rule adapt_compare. Globs every flood_metrics.csv under the basin's
runs/ folder, not only this invocation's targets: snakemake.input only
triggers reruns, and deleted strategy folders drop out of the table.
"""

from pathlib import Path

import pandas as pd

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

# comparison_csv sits directly in the basin's runs/ folder.
runs_root = Path(snakemake.output.comparison_csv).parent


def _read_labeled(paths: list[Path], method: str, strategy_from_path: bool) -> pd.DataFrame:
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
