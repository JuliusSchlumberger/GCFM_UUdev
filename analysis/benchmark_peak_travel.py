"""Benchmark rule 11's walled peak travel time against the crest-calibration SFINCS run.

Rule 11 holds the sea at the coastal protection storm tide from the start
(zsini no lower than it) and ramps only the rivers, so every water-level
change in its calibration runs is the river's. Per forcing point, along the
mainstem rule 11 sizes its run on, in round 1 (confined by 1000 m walls: the
walled case itself; round 2 matches it wherever no production crest is
overtopped):

  observed: t50 of the water level at each node (first time its rise from the
  run start reaches half its rise to the run end) minus t50 of the forcing
  discharge (half its ramp);
  estimate: src.river_profile.walled_travel_time from the forcing node, with
  rule 11's inputs (round-0 bed, q_calib per reach, the highest station's
  storm tide, channel_manning_n, ramp_hr, wave_regime_rT): t_wave_hr (rule
  11's run length), t_kinematic_hr and t_gravity_hr.

Compared at the mouth, the path's last node with a rise of at least
MIN_RISE_M. A run whose sea boundary changes over time is refused. tau95 (hours
before the peak the inflow stays within 5 % of it) of the RP100 GloFAS event at
the main inflow sets the scale an error matters on for the compound lag.

Reads rule 11's outputs (node table, depth-estimated network,
depth_crest_calibration/round1/), river_forcing.nc and config; writes
<results_dir>/figures/analysis/benchmark_peak_travel_summary.csv (one row per
basin and forcing point) and, per basin, <results_dir>/<basin>/figures/analysis/
90_benchmark_peak_travel_nodes.csv and _benchmark.png.

Usage:
    python analysis/benchmark_peak_travel.py            # every basin with a calibration run
    python analysis/benchmark_peak_travel.py 2433835 3279946
"""

import logging

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from _upstream_profile import mainstem_profile
from check_event_timing import hours_before_peak_within, read_inp
from check_river_bed import _config, _git_hash

log = logging.getLogger(__name__)

EVENT_RP_YR = 100.0
MIN_RISE_M = 0.05
SEA_HELD_TOL_M = 0.01


def arrival_hr(t_hr: np.ndarray, series: np.ndarray, frac: float = 0.5) -> float:
    """Return the first time (h, linear between samples) the rise from series[0] reaches frac of the rise to series[-1].

    NaN when the total rise is below MIN_RISE_M (in the series' units) or the
    series is not finite at both ends.
    """
    s = np.asarray(series, float)
    if not (np.isfinite(s[0]) and np.isfinite(s[-1])) or s[-1] - s[0] < MIN_RISE_M:
        return float("nan")
    target = s[0] + frac * (s[-1] - s[0])
    k = int(np.argmax(np.nan_to_num(s, nan=-np.inf) >= target))
    if k == 0:
        return float(t_hr[0])
    return float(np.interp(target, [s[k - 1], s[k]], [t_hr[k - 1], t_hr[k]]))


def _map_series(
    map_path, x: np.ndarray, y: np.ndarray, inp: dict
) -> tuple[np.ndarray, np.ndarray]:
    if float(inp.get("rotation", 0.0)) != 0.0:
        raise ValueError(f"{map_path}: rotated grid not supported")
    x0, y0, dx, dy = (float(inp[k]) for k in ("x0", "y0", "dx", "dy"))
    with xr.open_dataset(map_path, decode_times=False) as ds:
        units = ds["time"].attrs.get("units", "seconds")
        if not units.startswith("seconds"):
            raise ValueError(
                f"{map_path}: time units {units!r}, expected seconds since tref"
            )
        t_hr = ds["time"].values.astype(float) / 3600.0
        n = np.floor((y - y0) / dy).astype(int)
        m = np.floor((x - x0) / dx).astype(int)
        ok = (n >= 0) & (n < ds.sizes["n"]) & (m >= 0) & (m < ds.sizes["m"])
        zs = np.full((len(t_hr), len(x)), np.nan)
        zs[:, ok] = ds["zs"].values[:, n[ok], m[ok]]
    return t_hr, zs


def _event_tau95(river_forcing_path, event_cfg: dict) -> float:
    from src.river_forcing import build_design_discharge_matrix

    with xr.open_dataset(river_forcing_path, decode_times=False) as ds:
        ds = ds.load()
    active = ds["has_glofas"].values.astype(bool)
    q = build_design_discharge_matrix(ds, active, EVENT_RP_YR, event=event_cfg)
    main = int(np.nanargmax(ds["bankfull_discharge_m3s"].values[active]))
    return hours_before_peak_within(ds["time"].values.astype(float), q[main], 0.95)


def benchmark_basin(results_dir, basin_id: str, cfg: dict) -> pd.DataFrame | None:
    from src.river_network import build_downstream_adjacency, normalize_reach_id
    from src.river_profile import walled_travel_time

    pre = results_dir / basin_id / "preprocessing_inputs"
    run = pre / "depth_crest_calibration" / "round1"
    nodes_path = (
        pre
        / "visuals"
        / "11_calibration"
        / "river_bed"
        / f"{basin_id}_river_bed_nodes.csv"
    )
    if not (run / "sfincs_map.nc").exists() or not nodes_path.exists():
        log.info(f"{basin_id}: skipped, no calibration run or node table")
        return None
    bzs = np.loadtxt(run / "sfincs.bzs", ndmin=2)
    if np.abs(bzs[:, 1:] - bzs[0, 1:]).max() > SEA_HELD_TOL_M:
        log.info(
            f"{basin_id}: skipped, the calibration sea boundary changes over the run "
            f"(run predates the held-sea setup) -- rerun rule modelled_depth_estimation"
        )
        return None

    rdm = cfg["river_processing"]["river_depth_modelling"]
    manning_n = float(cfg["river_processing"]["channel_manning_n"])
    nodes = pd.read_csv(nodes_path)
    nodes["reach_id"] = nodes["reach_id"].map(normalize_reach_id)
    rivers = gpd.read_file(
        pre / "domain" / f"{basin_id}_river_network_depth_estimated.gpkg"
    )
    downstream = build_downstream_adjacency(rivers)
    q_calib = dict(
        zip(rivers["reach_id"].map(normalize_reach_id), rivers["q_calib"].astype(float))
    )
    tau95_hr = _event_tau95(
        pre / "forcing" / "river_forcing.nc",
        cfg["boundary_forcings"]["river"]["event_hydrograph"],
    )

    inp = read_inp(run / "sfincs.inp")
    src = np.loadtxt(run / "sfincs.src", usecols=(0, 1), ndmin=2)
    dis = np.loadtxt(run / "sfincs.dis", ndmin=2)
    sea_level_m = float(bzs[0, 1:].max())
    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()

    rows, node_tables = [], []
    for k, (sx, sy) in enumerate(src):
        reach = nodes["reach_id"].iloc[int(np.argmin(np.hypot(nx - sx, ny - sy)))]
        prof = mainstem_profile(nodes, downstream, reach, reach_q=q_calib)
        start = int(np.argmin(np.hypot(prof["x"] - sx, prof["y"] - sy)))
        prof = prof.iloc[start:].reset_index(drop=True)
        est = walled_travel_time(
            prof,
            prof["reach_id"].map(q_calib).to_numpy(float),
            sea_level_m,
            manning_n,
            forcing_hr=float(rdm["ramp_hr"]),
            regime_rT=tuple(float(v) for v in rdm["wave_regime_rT"]),
        )
        t_hr, zs = _map_series(
            run / "sfincs_map.nc", prof["x"].to_numpy(), prof["y"].to_numpy(), inp
        )
        t_force_hr = arrival_hr(dis[:, 0] / 3600.0, dis[:, 1 + k])
        t_obs = (
            np.array([arrival_hr(t_hr, zs[:, i]) for i in range(len(prof))])
            - t_force_hr
        )
        valid = np.flatnonzero(np.isfinite(t_obs))
        if valid.size == 0:
            log.info(f"{basin_id} forcing point {k}: no node rises by {MIN_RISE_M} m")
            continue
        mouth = int(valid[-1])
        node_tables.append(
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "forcing_point": k,
                    "reach_id": prof["reach_id"],
                    "dist_from_mouth_m": prof["dist_from_mouth_m"],
                    "t_obs_hr": t_obs,
                    **{
                        c: est[c]
                        for c in (
                            "t_wave_hr",
                            "t_kinematic_hr",
                            "t_gravity_hr",
                            "regime",
                        )
                    },
                }
            )
        )
        rows.append(
            {
                "basin_id": basin_id,
                "forcing_point": k,
                "q_forcing_m3s": float(dis[-1, 1 + k]),
                "path_km": float(prof["dist_from_mouth_m"].iloc[0]) / 1e3,
                "mouth_km": float(prof["dist_from_mouth_m"].iloc[mouth]) / 1e3,
                "rise_at_mouth_m": float(zs[-1, mouth] - zs[0, mouth]),
                "t_obs_hr": float(t_obs[mouth]),
                "t_wave_hr": float(est["t_wave_hr"].iloc[mouth]),
                "t_gravity_hr": float(est["t_gravity_hr"].iloc[mouth]),
                "error_hr": float(est["t_wave_hr"].iloc[mouth] - t_obs[mouth]),
                "error_share": float(est["t_wave_hr"].iloc[mouth] / t_obs[mouth] - 1.0),
                "run_hr": float(t_hr[-1]),
                "tau95_event_hr": tau95_hr,
            }
        )
    if not rows:
        return None

    nodes_out = pd.concat(node_tables, ignore_index=True)
    nodes_out.to_csv(
        analysis_path(basin_id, __file__, "nodes.csv"),
        index=False,
        float_format="%.4f",
    )
    df = pd.DataFrame(rows)
    _plot(nodes_out, df, analysis_path(basin_id, __file__, "benchmark.png"))
    return df


def _plot(nodes: pd.DataFrame, df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(len(df), 1, figsize=(8, 3.6 * len(df)), squeeze=False)
    for ax, (_, r) in zip(axes[:, 0], df.iterrows()):
        t = nodes[nodes["forcing_point"] == r["forcing_point"]]
        km = t["dist_from_mouth_m"] / 1e3
        ax.plot(
            km,
            t["t_wave_hr"],
            color="k",
            lw=1.2,
            label="estimate (rule 11 walled travel time)",
        )
        ax.plot(
            km,
            t["t_gravity_hr"],
            color="tab:green",
            lw=1.0,
            ls="--",
            label="fastest possible (gravity wave)",
        )
        ax.plot(km, t["t_obs_hr"], "o", ms=3, color="tab:blue", label="SFINCS round 1")
        ax.set_title(
            f"forcing point {r['forcing_point']}: {r['q_forcing_m3s']:.0f} m3/s; error at the mouth "
            f"{r['error_hr']:+.1f} h; event tau95 {r['tau95_event_hr']:.1f} h",
            fontsize=9,
        )
        ax.set_ylabel(
            "arrival of the river high water (h)\n(half-way rise after the inflow's)"
        )
        ax.set_xlim(r["path_km"], 0.0)
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("distance from mouth (km)")
    fig.suptitle(
        f"{df['basin_id'].iloc[0]} walled travel vs calibration run ({_git_hash()})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        help="basin ids (default: every basin with a calibration run)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = site_path("results_dir")
    cfg = _config()
    basin_ids = args.basin_ids or sorted(
        p.name for p in results_dir.iterdir() if p.name.isdigit()
    )
    tables = [
        t for b in basin_ids if (t := benchmark_basin(results_dir, b, cfg)) is not None
    ]
    if not tables:
        raise SystemExit(
            f"No basin with a held-sea crest-calibration run under {results_dir}"
        )
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(
        analysis_path(None, __file__, "summary.csv"), index=False, float_format="%.2f"
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
