"""Where a compound lag puts the surge peak on the river's discharge plateau at the main mouth.

From the same coastal-only and river-only runs as analysis/peak_lag.py, per
basin, at the main mouth (largest tide-filtered river discharge):

  plateau: the span where the Godin-filtered mouth discharge stays within
  --plateau-frac of its maximum; open-ended when it still holds at the edge of
  the filter's valid window (the run ends before the plateau does);
  lag_mid_hr = surge_travel_hr - (plateau midpoint - t_river_inflow), the lag
  that puts the surge peak at the plateau's midpoint. t_river_inflow is the
  unfiltered inflow peak, the time rule 14 keeps the river peak at.

The figure has one row per basin: inflow and mouth discharge with the
plateau shaded (left), and the coastal run's mouth level shifted to where the
configured lag and lag_mid_hr put it in a compound run (right).

Writes lag_plateau_<coast>_<river>.png and .csv to <results_dir>/figures/analysis/.

Usage: python analysis/lag_plateau.py [basin_id ...] [--coast coast_100_03SLR] [--river river_100_03SLR] [--plateau-frac 0.98]
"""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from check_river_bed import _git_hash
from peak_lag import DEFAULT_BASINS, _his, _table, godin, peak_hr


def configured_lag_hr(basin_id: str) -> float:
    with open(REPO_ROOT / "config" / "config.yml", encoding="utf-8") as f:
        compound = yaml.safe_load(f)["sfincs"]["boundary_setup"]["compound"]
    overrides = {str(b): h for b, h in (compound.get("lag_overrides_hr") or {}).items()}
    return float(overrides.get(basin_id, compound["lag_hr"]))


def plateau_basin(
    results_dir, basin_id: str, coast: str, river: str, frac: float
) -> dict:
    root = results_dir / basin_id
    mouths = gpd.read_file(root / "sfincs_skeleton" / f"{basin_id}_mouths.gpkg")
    coast_dir = root / "runs" / coast / "sfincs"
    river_dir = root / "runs" / river / "sfincs"

    t_bzs, bzs = _table(coast_dir, "sfincs.bzs")
    t_surge_boundary = peak_hr(t_bzs, bzs.mean(axis=1))
    t_dis, dis = _table(river_dir, "sfincs.dis")
    q_in = dis.sum(axis=1)
    t_river_inflow = peak_hr(t_dis, q_in)

    cds, tc = _his(coast_dir)
    rds, tr = _his(river_dir)
    dt_r = float(np.median(np.diff(tr)))
    crs_names = [
        str(n).strip() for n in np.asarray(rds["crosssection_name"].values).astype(str)
    ]
    best = None
    for _, m in mouths.iterrows():
        # Seaward discharge is negative at the mouth cross-sections; filter the
        # signed series so tidal reversals cancel instead of rectifying.
        q = -rds["crosssection_discharge"].values[:, crs_names.index(m["name"])]
        q_f = godin(q, dt_r)
        if best is None or np.nanmax(q_f) > np.nanmax(best[2]):
            best = (m, q, q_f, cds["point_zs"].values[:, int(m["obs_index"])])
    cds.close()
    rds.close()
    m, q, q_f, zs_coast = best

    valid = np.isfinite(q_f)
    q_max = np.nanmax(q_f)
    on = valid & (q_f >= frac * q_max)
    start, end = float(tr[on][0]), float(tr[on][-1])
    open_ended = end >= float(tr[valid][-1])
    mid = 0.5 * (start + end)
    t_surge_mouth = peak_hr(tc, zs_coast)
    surge_travel = t_surge_mouth - t_surge_boundary
    lag_cfg = configured_lag_hr(basin_id)
    return {
        "basin_id": basin_id,
        "mouth": m["name"],
        "t_river_inflow_hr": t_river_inflow,
        "q_inflow_peak_m3s": float(q_in.max()),
        "t_mouth_q_max_hr": peak_hr(tr, q_f),
        "q_mouth_max_m3s": float(q_max),
        "plateau_start_hr": start,
        "plateau_end_hr": end,
        "plateau_open_ended": open_ended,
        "plateau_mid_hr": mid,
        "surge_travel_hr": surge_travel,
        "lag_mid_hr": surge_travel - (mid - t_river_inflow),
        "lag_config_hr": lag_cfg,
        "_series": (t_dis, q_in, tr, q, q_f, tc, zs_coast, t_surge_mouth),
    }


def _surge_mouth_peak_hr(r: dict, lag_hr: float) -> float:
    return r["t_river_inflow_hr"] - lag_hr + r["surge_travel_hr"]


def plot(rows: list[dict], coast: str, river: str, frac: float, tag: str, path) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.8 * len(rows)), squeeze=False)
    for (ax_q, ax_s), r in zip(axes, rows):
        t_dis, q_in, tr, q, q_f, tc, zs_c, t_sm = r["_series"]
        end_lbl = "open" if r["plateau_open_ended"] else f"{r['plateau_end_hr']:.0f}"
        band = dict(color="tab:green", alpha=0.15)
        lo, hi = t_dis[0], t_dis[-1]

        ax_q.plot(t_dis, q_in, color="0.3", lw=1.2, label="inflow (sum, sfincs.dis)")
        ax_q.plot(
            tr, q, color="tab:orange", lw=0.5, alpha=0.5, label="main mouth discharge"
        )
        ax_q.plot(
            tr, q_f, color="tab:red", lw=1.6, label="main mouth, tide-filtered (Godin)"
        )
        ax_q.axvspan(
            r["plateau_start_hr"],
            r["plateau_end_hr"],
            **band,
            label=f"plateau >= {frac:.0%} of max: {r['plateau_start_hr']:.0f}-{end_lbl} h",
        )
        ax_q.axvline(r["t_river_inflow_hr"], color="0.3", ls=":", lw=1)
        ax_q.axvline(r["plateau_mid_hr"], color="tab:green", ls="--", lw=1)
        ax_q.set_xlim(lo, hi)
        ax_q.set_ylabel("discharge (m3/s)")
        ax_q.set_title(
            f"{r['basin_id']} {r['mouth']}: inflow peak {r['t_river_inflow_hr']:.0f} h "
            f"({r['q_inflow_peak_m3s']:.0f} m3/s), mouth max {r['q_mouth_max_m3s']:.0f} m3/s "
            f"at {r['t_mouth_q_max_hr']:.0f} h",
            fontsize=9,
        )
        ax_q.legend(fontsize=7, loc="upper left")
        ax_q.grid(alpha=0.3)

        ax_s.axvspan(
            r["plateau_start_hr"], r["plateau_end_hr"], **band, label="river plateau"
        )
        for lag, color, name in (
            (r["lag_config_hr"], "tab:blue", "config"),
            (r["lag_mid_hr"], "tab:purple", "plateau midpoint"),
        ):
            t_peak = _surge_mouth_peak_hr(r, lag)
            ax_s.plot(
                tc + (t_peak - t_sm),
                zs_c,
                color=color,
                lw=1.2,
                label=f"surge at mouth, lag {lag:+.0f} h ({name}): peak {t_peak:.0f} h",
            )
            ax_s.axvline(t_peak, color=color, ls="--", lw=1)
        ax_s.axvline(
            r["t_river_inflow_hr"], color="0.3", ls=":", lw=1, label="inflow peak"
        )
        ax_s.set_xlim(lo, hi)
        ax_s.set_ylabel("water level at mouth (m)")
        ax_s.set_title(
            f"surge travel {r['surge_travel_hr']:.1f} h; lag = river peak - surge peak at the boundary",
            fontsize=9,
        )
        ax_s.legend(fontsize=7, loc="upper left")
        ax_s.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("hours from tref")
    fig.suptitle(
        f"surge peak vs river plateau at the main mouth | {coast} vs {river} | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument("--river", default="river_100_03SLR")
    parser.add_argument("--plateau-frac", type=float, default=0.98)
    args = parser.parse_args()
    results_dir = site_path("results_dir")
    rows = [
        plateau_basin(results_dir, str(b), args.coast, args.river, args.plateau_frac)
        for b in args.basin_ids
    ]
    stem = f"{args.coast}_{args.river}"
    plot(
        rows,
        args.coast,
        args.river,
        args.plateau_frac,
        _git_hash(),
        analysis_path(None, __file__, f"{stem}.png"),
    )
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_series"} for r in rows])
    df.to_csv(
        analysis_path(None, __file__, f"{stem}.csv"), index=False, float_format="%.2f"
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
