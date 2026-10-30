"""Phase 1.0: wall-clock per rule of a Snakemake run, and the flood extent and volume it produced.

Parses a .snakemake/log/*.snakemake.log ("[date]" + "rule X:" ... "jobid: N"
start lines and "Finished jobid: N (Rule: X)") into the duration of every job,
and reads runs/<scenario>/metrics/flood_metrics.csv of the basin.

Writes 90_phase1_run_times_<label>.csv to <results_dir>/<basin>/figures/analysis/
and appends a row to phase1_run_times_summary.csv in <results_dir>/figures/analysis/.

Usage: python analysis/phase1_run_times.py LOG basin_id [--scenario default] [--label baseline]
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import basin_paths

STAMP = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]")
START = re.compile(r"^(?:local)?rule (\w+):")
JOBID = re.compile(r"^\s+jobid: (\d+)")
DONE = re.compile(r"^Finished jobid: (\d+) \(Rule: (\w+)\)")


def parse_log(path: Path) -> pd.DataFrame:
    """Return one row per finished job: rule, start, end, minutes."""
    stamp, pending_rule, starts, rows = None, None, {}, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if m := STAMP.match(line):
            stamp = datetime.strptime(
                " ".join(m.group(1).split()), "%a %b %d %H:%M:%S %Y"
            )
        elif m := START.match(line):
            pending_rule = (m.group(1), stamp)
        elif (m := JOBID.match(line)) and pending_rule is not None:
            starts[m.group(1)] = pending_rule
            pending_rule = None
        elif m := DONE.match(line):
            rule, t0 = starts.get(m.group(1), (m.group(2), None))
            rows.append(
                {
                    "rule": rule,
                    "start": t0,
                    "end": stamp,
                    "minutes": (stamp - t0).total_seconds() / 60.0
                    if t0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=["rule", "start", "end", "minutes"])


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("basin_id")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    jobs = parse_log(args.log)
    jobs.to_csv(
        analysis_path(args.basin_id, __file__, f"{args.label}.csv"),
        index=False,
        float_format="%.2f",
    )
    metrics_csv = (
        basin_paths(args.basin_id)["root"]
        / "runs"
        / args.scenario
        / "metrics"
        / "flood_metrics.csv"
    )
    metrics = pd.read_csv(metrics_csv).iloc[0].to_dict() if metrics_csv.exists() else {}
    row = {
        "basin_id": args.basin_id,
        "label": args.label,
        "log": args.log.name,
        "n_jobs": len(jobs),
        "total_minutes": (jobs["end"].max() - jobs["start"].min()).total_seconds()
        / 60.0
        if len(jobs)
        else float("nan"),
        "flooded_area_km2": metrics.get("flooded_area_km2"),
        "volume_m3": metrics.get("volume_m3"),
        "flood_extent_pct": metrics.get("flood_extent_pct"),
        **{f"min_{r}": m for r, m in jobs.groupby("rule")["minutes"].sum().items()},
    }
    out = analysis_path(None, __file__, "summary.csv")
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    old = (
        old[
            ~(
                (old.get("basin_id", pd.Series(dtype=str)).astype(str) == args.basin_id)
                & (old.get("label", pd.Series(dtype=str)) == args.label)
            )
        ]
        if len(old)
        else old
    )
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
        out, index=False, float_format="%.4g"
    )
    print(jobs.to_string(index=False))
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
