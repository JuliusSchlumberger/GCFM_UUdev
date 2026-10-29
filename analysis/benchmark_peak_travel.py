"""Benchmark rule 11's walled peak travel time against the crest-calibration SFINCS run.

Per forcing point, along the mainstem rule 11 sizes its run on, in round 1
(confined by 1000 m walls: the walled case itself; round 2 matches it wherever
no production crest is overtopped):

  observed: the settle time of the water level at each node -- searching back
  from the run end, the last time it is more than steady_check.tol_m from its
  final level -- minus ramp_hr, when the full calibration discharge enters;
  estimate: src.river_profile.walled_travel_time from the forcing node, with
  rule 11's inputs (round-0 bed, q_calib per reach, the highest final sea
  level, channel_manning_n, ramp_hr, wave_regime_rT): t_wave_hr (rule 11's run
  length), t_kinematic_hr and t_gravity_hr.

The sea ramps over the same ramp_hr and then holds, and crosses the delta in
well under an hour, so a change after ramp_hr + SEA_SETTLE_HR is the river's.
A node whose settle time falls within that is sea-limited: the river changes
it by less than tol_m after the sea has settled, too little to matter for the
lag. A node still changing within steady_check.window_hr of the run end is
unsettled and its settle time a lower bound. Settling includes the slow
backwater adjustment, so it can trail the wave front the estimate describes.
tau95 (hours before the peak the inflow stays within 5 % of it) of the RP100
GloFAS event at the main inflow sets the scale an error matters on for the
compound lag.

Reads rule 11's outputs (node table, depth-estimated network,
depth_crest_calibration/round1/), river_forcing.nc and config; writes
figs/peak_travel_benchmark/benchmark.csv (one row per basin and forcing point),
<basin>_nodes.csv and <basin>_benchmark.png.

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
from _common import REPO_ROOT, script_parser, site_path
from _upstream_profile import mainstem_profile
from check_event_timing import hours_before_peak_within, read_inp
from check_river_bed import _config, _git_hash

log = logging.getLogger(__name__)

EVENT_RP_YR = 100.0
SEA_SETTLE_HR = 1.0


def settle_hr(t_hr: np.ndarray, series: np.ndarray, tol: float) -> float:
    """Return the last time (h, linear between samples) the series is more than tol from its final value.

    NaN when it never is, or when the final value is not finite.
    """
    s = np.asarray(series, float)
    if not np.isfinite(s[-1]):
        return float("nan")
    off = np.abs(s - s[-1]) > tol
    if not off.any():
        return float("nan")
    k = int(np.flatnonzero(off)[-1])
    d0, d1 = abs(s[k] - s[-1]), abs(s[k + 1] - s[-1])
    return float(np.interp(tol, [d1, d0], [t_hr[k + 1], t_hr[k]]))


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


def benchmark_basin(
    results_dir, basin_id: str, cfg: dict, out_dir
) -> pd.DataFrame | None:
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

    rdm = cfg["river_processing"]["river_depth_modelling"]
    manning_n = float(cfg["river_processing"]["channel_manning_n"])
    ramp_hr = float(rdm["ramp_hr"])
    tol_m = float(rdm["steady_check"]["tol_m"])
    window_hr = float(rdm["steady_check"]["window_hr"])
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
    sea_peak_m = float(np.loadtxt(run / "sfincs.bzs", ndmin=2)[-1, 1:].max())
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
            sea_peak_m,
            manning_n,
            forcing_hr=ramp_hr,
            regime_rT=tuple(float(v) for v in rdm["wave_regime_rT"]),
        )
        t_hr, zs = _map_series(
            run / "sfincs_map.nc", prof["x"].to_numpy(), prof["y"].to_numpy(), inp
        )
        settle = np.array([settle_hr(t_hr, zs[:, i], tol_m) for i in range(len(prof))])
        river_driven = settle > ramp_hr + SEA_SETTLE_HR
        unsettled = settle > t_hr[-1] - window_hr
        t_obs = settle - ramp_hr
        end = int(np.flatnonzero(river_driven)[-1]) if river_driven.any() else 0
        node_tables.append(
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "forcing_point": k,
                    "reach_id": prof["reach_id"],
                    "dist_from_mouth_m": prof["dist_from_mouth_m"],
                    "t_obs_hr": t_obs,
                    "river_driven": river_driven,
                    "unsettled": unsettled,
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
                "river_driven_to_km": float(prof["dist_from_mouth_m"].iloc[end]) / 1e3,
                "t_obs_hr": float(t_obs[end]),
                "t_wave_hr": float(est["t_wave_hr"].iloc[end]),
                "t_gravity_hr": float(est["t_gravity_hr"].iloc[end]),
                "error_hr": float(est["t_wave_hr"].iloc[end] - t_obs[end]),
                "error_share": float(est["t_wave_hr"].iloc[end] / t_obs[end] - 1.0),
                "t_wave_mouth_hr": float(est["t_wave_hr"].iloc[-1]),
                "n_unsettled": int(unsettled.sum()),
                "run_hr": float(t_hr[-1]),
                "tau95_event_hr": tau95_hr,
            }
        )

    nodes_out = pd.concat(node_tables, ignore_index=True)
    nodes_out.to_csv(
        out_dir / f"{basin_id}_nodes.csv", index=False, float_format="%.4f"
    )
    df = pd.DataFrame(rows)
    _plot(nodes_out, df, out_dir / f"{basin_id}_benchmark.png")
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
        rd, un = t["river_driven"].astype(bool), t["unsettled"].astype(bool)
        ax.plot(
            km[rd & ~un],
            t["t_obs_hr"][rd & ~un],
            "o",
            ms=3,
            color="tab:blue",
            label="SFINCS: river settled",
        )
        ax.plot(
            km[un],
            t["t_obs_hr"][un],
            "o",
            ms=3,
            mfc="none",
            color="tab:red",
            label="SFINCS: still changing at run end",
        )
        ax.plot(
            km[~rd],
            t["t_obs_hr"][~rd],
            "o",
            ms=3,
            color="0.6",
            label="SFINCS: settled with the sea (river < tol)",
        )
        ax.set_title(
            f"forcing point {r['forcing_point']}: {r['q_forcing_m3s']:.0f} m3/s; error at the last river-driven node "
            f"{r['error_hr']:+.1f} h; event tau95 {r['tau95_event_hr']:.1f} h",
            fontsize=9,
        )
        ax.set_ylabel(
            "hours after the full discharge enters\nuntil the level stops changing"
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
    out_dir = REPO_ROOT / "figs" / "peak_travel_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    basin_ids = args.basin_ids or sorted(
        p.name for p in results_dir.iterdir() if p.name.isdigit()
    )
    tables = [
        t
        for b in basin_ids
        if (t := benchmark_basin(results_dir, b, cfg, out_dir)) is not None
    ]
    if not tables:
        raise SystemExit(f"No basin with a crest-calibration run under {results_dir}")
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out_dir / "benchmark.csv", index=False, float_format="%.2f")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
