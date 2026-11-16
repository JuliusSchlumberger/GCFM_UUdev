"""How far up each forcing mainstem a coastal event raises the river level, and when, from a finished coastal run.

Per forcing point, along its mainstem (largest Q_bf at bifurcations, as
check_upstream_boundary), the SFINCS level of the active cell nearest each
river node in the coastal run (river held at mean discharge) against the
spin-up's end level (same discharge, calm sea, the event's restart):

  dz = zsmax over the run - spin-up level: tide and surge together (the
       storm-tide hydrograph spans the whole run, so no stretch is tide only);
  x_inf: first node, going upstream, where dz < max(level_tol_m,
         amplitude_threshold_fraction x dz at the mouth)
         (testing.upstream_boundary_check), and the forcing point's verdict
         with its margin_km;
  t_peak - t_surge: hour of each node's maximum level minus the hour of the
         peak of the mean sea boundary (map output interval).

The steady check (check_upstream_boundary.py) holds the sea at its most
extreme level; this one is the dynamic reach of one finite event.

Writes 90_surge_reach_<scenario>_profiles.csv and _profiles.png to
<results_dir>/<basin>/figures/analysis/ and surge_reach_<scenario>_summary.csv
to <results_dir>/figures/analysis/.

Usage: python analysis/surge_reach.py [basin_id ...] [--scenario coast_100_03SLR]
"""

import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _basin_inputs import spinup_dir
from _common import REPO_ROOT, analysis_dir, analysis_path, script_parser, site_path
from _upstream_profile import (
    boundary_verdict,
    influence_length,
    mainstem_path,
    path_nodes,
    routed_discharge,
)
from check_river_bed import _config, _git_hash, run_round0
from check_upstream_boundary import forcing_points
from scipy.spatial import cKDTree

sys.path.insert(0, str(REPO_ROOT / "workflow"))

log = logging.getLogger(__name__)


def _hours(ds: xr.Dataset, dim: str) -> np.ndarray:
    t = ds[dim].values
    if np.issubdtype(t.dtype, np.datetime64):
        return (t - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "h")
    return t.astype(float) / 3600.0


def sample_frames(map_path, x: np.ndarray, y: np.ndarray, var: str):
    """Return ``var`` (time, point) at the active cell nearest each point, its hours from tref and the cell distance."""
    with xr.open_dataset(map_path) as ds:
        active = ds["msk"].values > 0
        n_idx, m_idx = np.nonzero(active)
        dist, k = cKDTree(np.c_[ds["x"].values[active], ds["y"].values[active]]).query(
            np.c_[x, y]
        )
        n_sel, m_sel = n_idx[k], m_idx[k]
        da = ds[var]
        tdim = [d for d in da.dims if d not in ("n", "m")][0]
        # one frame at a time: pointwise indexing of the compressed array is slow
        out = np.vstack(
            [
                np.asarray(da.isel({tdim: i}).values)[n_sel, m_sel]
                for i in range(da.sizes[tdim])
            ]
        )
        return out, _hours(ds, tdim), dist


def reach_basin(basin_id: str, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _config()
    test = cfg["testing"]["upstream_boundary_check"]
    level_tol_m = float(test["level_tol_m"])
    frac = float(test["amplitude_threshold_fraction"])
    margin_m = float(test["margin_km"]) * 1000
    root = site_path("results_dir") / basin_id
    run = root / "runs" / scenario / "sfincs"
    frc = root / "preprocessing_inputs" / "forcing"

    bed, rivers, _forcing, _sea, crs, _paths = run_round0(
        basin_id, cfg, analysis_dir(basin_id)
    )
    net = bed.net
    q_bf = routed_discharge(rivers, frc / "river_forcing.nc", "bankfull_discharge_m3s")

    bzs = np.loadtxt(run / "sfincs.bzs")
    # bzs times are seconds from tref
    t_surge_hr = float(bzs[np.argmax(bzs[:, 1:].mean(axis=1)), 0] / 3600.0)
    log.info(f"{basin_id}: sea boundary peak {t_surge_hr:.1f} h")

    rows, prof_rows = [], []
    paths = []
    for p in forcing_points(
        frc / "river_forcing.nc", rivers, crs, net, []
    ).itertuples():
        if p.reach_id not in net.reach_nodes:
            log.warning(
                f"Forcing reach {p.reach_id} has no nodes in the network: skipped"
            )
            continue
        ids, dist = path_nodes(net, mainstem_path(net.downstream, p.reach_id, q_bf))
        k_f = int(np.argmin(np.hypot(net.node_x[ids] - p.x, net.node_y[ids] - p.y)))
        paths.append((p.reach_id, ids[k_f:], dist[k_f:]))
    nodes = np.unique(np.concatenate([ids for _, ids, _ in paths]))
    nx, ny = net.node_x[nodes], net.node_y[nodes]

    zs, t_hr, cell_d = sample_frames(run / "sfincs_map.nc", nx, ny, "zs")
    zs_spin, _, _ = sample_frames(
        spinup_dir(basin_id, scenario) / "sfincs_map.nc", nx, ny, "zs"
    )
    ref = zs_spin[-1]
    with xr.open_dataset(run / "sfincs_map.nc") as ds:
        zsmax_all = ds["zsmax"].max(dim=ds["zsmax"].dims[0]).values
        active = ds["msk"].values > 0
        n_idx, m_idx = np.nonzero(active)
        _, k = cKDTree(np.c_[ds["x"].values[active], ds["y"].values[active]]).query(
            np.c_[nx, ny]
        )
        zsmax = zsmax_all[n_idx[k], m_idx[k]]
    dz_all = zsmax - ref
    t_peak = t_hr[np.nanargmax(np.where(np.isfinite(zs), zs, -np.inf), axis=0)]
    pos = {int(n): i for i, n in enumerate(nodes)}

    for reach_id, ids, dist in paths:
        j = np.array([pos[int(n)] for n in ids])
        x_f = float(dist[0])
        dz = dz_all[j]
        a0 = float(dz[-1])
        thr = max(level_tol_m, frac * a0)
        x_inf = influence_length(dist, dz, thr)
        ok, why = boundary_verdict(x_f, float(dz[0]), x_inf, level_tol_m, margin_m)
        lag = t_peak[j] - t_surge_hr
        reached = dz >= thr
        rows.append(
            {
                "basin_id": basin_id,
                "scenario": scenario,
                "reach_id": reach_id,
                "x_f_km": x_f / 1000,
                "t_surge_boundary_hr": t_surge_hr,
                "a0_m": a0,
                "threshold_m": thr,
                "x_inf_km": x_inf / 1000,
                "margin_km": (x_f - x_inf) / 1000,
                "dz_at_f_m": float(dz[0]),
                "passes": ok,
                "deciding": why,
                "lag_mouth_hr": float(lag[-1]),
                "lag_at_x_inf_hr": float(np.nanmax(lag[reached]))
                if reached.any()
                else np.nan,
                "max_cell_distance_m": float(cell_d[j].max()),
            }
        )
        prof_rows.append(
            pd.DataFrame(
                {
                    "reach_id": reach_id,
                    "node": ids,
                    "dist_from_mouth_km": dist / 1000,
                    "zs_spinup_m": ref[j],
                    "zsmax_m": zsmax[j],
                    "dz_m": dz,
                    "t_peak_minus_surge_hr": lag,
                    "cell_distance_m": cell_d[j],
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(prof_rows, ignore_index=True)


def plot_basin(summary: pd.DataFrame, prof: pd.DataFrame, path, title: str) -> None:
    reaches = list(summary["reach_id"])
    fig, axes = plt.subplots(
        2,
        len(reaches),
        figsize=(max(5.5 * len(reaches), 10), 7),
        squeeze=False,
        sharex="col",
    )
    for c, r in enumerate(reaches):
        s = summary[summary["reach_id"] == r].iloc[0]
        p = prof[prof["reach_id"] == r]
        a = axes[0, c]
        a.plot(
            p["dist_from_mouth_km"],
            p["dz_m"],
            color="tab:blue",
            label="zsmax - spin-up end",
        )
        a.axhline(s["threshold_m"], color="tab:blue", lw=0.8, ls=":", label="threshold")
        if np.isfinite(s["x_inf_km"]):
            a.axvline(s["x_inf_km"], color="tab:red", lw=1, ls="--", label="x_inf")
        a.axvline(s["x_f_km"], color="k", lw=1.2, label="forcing point")
        a.set_ylim(-0.5, 4.5)
        a.set_ylabel("level above spin-up end (m)")
        verdict = "PASS" if s["passes"] else f"FAIL ({s['deciding']})"
        a.set_title(
            f"reach {r}: forcing {s['x_f_km']:.0f} km, x_inf {s['x_inf_km']:.0f} km, {verdict}",
            fontsize=9,
        )
        a.grid(alpha=0.3)
        if c == 0:
            a.legend(fontsize=8)
        b = axes[1, c]
        m = p["dz_m"] >= s["threshold_m"]
        b.plot(
            p["dist_from_mouth_km"][m],
            p["t_peak_minus_surge_hr"][m],
            ".",
            color="tab:blue",
            ms=3,
        )
        b.plot(
            p["dist_from_mouth_km"][~m],
            p["t_peak_minus_surge_hr"][~m],
            ".",
            color="0.7",
            ms=3,
        )
        b.axvline(s["x_f_km"], color="k", lw=1.2)
        b.set_ylim(-48, 48)
        b.set_xlabel("distance from mouth (km)")
        b.set_ylabel("node peak - sea boundary peak (h)")
        b.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=["2433835"])
    parser.add_argument("--scenario", default="coast_100_03SLR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tag = _git_hash()
    summaries = []
    for basin_id in map(str, args.basin_ids):
        summary, prof = reach_basin(basin_id, args.scenario)
        prof.to_csv(
            analysis_path(basin_id, __file__, f"{args.scenario}_profiles.csv"),
            index=False,
        )
        plot_basin(
            summary,
            prof,
            analysis_path(basin_id, __file__, f"{args.scenario}_profiles.png"),
            f"surge_reach: basin {basin_id}, {args.scenario}, coastal run vs spin-up end; "
            f"grey: below threshold ({tag})",
        )
        summaries.append(summary)
    table = pd.concat(summaries, ignore_index=True)
    csv = analysis_path(None, __file__, f"{args.scenario}_summary.csv")
    table.to_csv(csv, index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    print(f"Written: {csv}")


if __name__ == "__main__":
    main()
