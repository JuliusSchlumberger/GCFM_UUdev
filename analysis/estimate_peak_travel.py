"""Estimate the river and surge peak travel times to the mouth with 1D wave speeds, and the lag and run length they imply.

River leg: steady GVF level at the scenario's peak main inflow along its mainstem
(rule 11's river-bed node table, sea at the storm-tide peak level), then
sum dx / c from the forcing point to the mouth with three celerities per node:
fast u + sqrt(g h) (dynamic wave), slow 5/3 u (kinematic wave, wide Manning
channel), regime: fast where the bed lies below the calm sea level (drowned),
slow elsewhere. Sea leg: fastest wet path from the water-level boundary cells
(skeleton mask 2) to the mouth over the skeleton bed at the storm-tide peak
level, sum dx / sqrt(g h).

The forcing aligns the river peak at the forcing point with the surge peak at
the boundary, so the river peak reaches the mouth lag = T_river - T_surge
after the surge peak does. The lag is material when it exceeds tau95 (hours
before the inflow peak that the inflow is within 5 % of it). Run length, with
t = 0 at the surge peak: unshifted, the run must last T_river + AFTER_MARGIN_HR
after it; shifted by the lag, it must last T_surge + AFTER_MARGIN_HR after it
and start lag + tau50 before it (the shifted rising limb above half the peak).
Channel only: the whole peak flows in the channel, so above bankfull
(q_peak_over_bankfull > 1) the slow celerity is too fast (overbank storage slows
the wave 3-4x, Dykstra & Dzwonkowski 2020); floodplain filling is checked on
finished runs by check_event_timing.py.

Reads build inputs only (node table, river network, skeleton gis/, the
scenario's sfincs.inp/.dis/.src/.bzs/.bnd), no SFINCS output. Writes
figs/peak_travel/peak_travel.csv (one row per basin and celerity),
<basin>_<scenario>_path.csv (per node) and <basin>_<scenario>_peak_travel.png.

Usage:
    python analysis/estimate_peak_travel.py                 # every basin with a node table and a built scenario
    python analysis/estimate_peak_travel.py 2433835 --scenario default
"""

import logging

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from _common import REPO_ROOT, script_parser, site_path
from _upstream_profile import forward_levels, mainstem_profile, scaled_discharge
from check_event_timing import (
    G_MS2,
    gravity_wave_travel,
    hours_before_peak_within,
    parse_timestamp,
    read_inp,
)
from check_river_bed import _config, _git_hash

log = logging.getLogger(__name__)

AFTER_MARGIN_HR = (
    12.0  # same settling time check_event_timing.py allows after the surge peak
)
MIN_DEPTH_M = 0.1
CELERITIES = ("fast", "regime", "slow")


def river_leg(
    profile: pd.DataFrame,
    q_m3s: np.ndarray,
    sea_level_m: float,
    calm_sea_level_m: float,
    manning_n: float,
) -> pd.DataFrame:
    """Return per profile node the flood-stage hydraulics and the travel time (h) from the first node per celerity."""
    eta = forward_levels(profile, q_m3s, sea_level_m, manning_n)
    z = profile["bed_m"].to_numpy(float)
    w = profile["width_m"].to_numpy(float)
    q = np.broadcast_to(np.asarray(q_m3s, float), z.shape)
    h = np.maximum(eta - z, MIN_DEPTH_M)
    u = q / (w * h)
    drowned = z < calm_sea_level_m
    fast = u + np.sqrt(G_MS2 * h)
    slow = 5.0 / 3.0 * u
    celerity = {"fast": fast, "regime": np.where(drowned, fast, slow), "slow": slow}
    dx = -np.diff(profile["dist_from_mouth_m"].to_numpy(float))
    out = profile[
        ["node", "reach_id", "x", "y", "dist_from_mouth_m", "bed_m", "width_m"]
    ].copy()
    out = out.assign(q_m3s=q, level_m=eta, depth_m=h, velocity_ms=u, drowned=drowned)
    with np.errstate(divide="ignore"):
        for name, c in celerity.items():
            out[f"celerity_{name}_ms"] = c
            out[f"t_{name}_hr"] = (
                np.concatenate([[0.0], np.cumsum(dx / (0.5 * (c[:-1] + c[1:])))])
                / 3600.0
            )
    return out


def sea_leg(
    bed_m: np.ndarray,
    mask: np.ndarray,
    sea_level_m: float,
    mouth_rc: tuple[int, int],
    dx_m: float,
) -> tuple[float, float]:
    """Return (hours, km) of the fastest wet path from the water-level boundary cells to the mouth cell.

    The mouth snaps to the nearest wet cell when its own cell is dry at ``sea_level_m``.
    """
    depth = sea_level_m - bed_m
    wet = (mask > 0) & np.isfinite(bed_m) & (depth > MIN_DEPTH_M)
    if not wet.any():
        return np.nan, np.nan
    if not wet[mouth_rc]:
        r, c = np.nonzero(wet)
        k = int(np.argmin((r - mouth_rc[0]) ** 2 + (c - mouth_rc[1]) ** 2))
        mouth_rc = (int(r[k]), int(c[k]))
    return gravity_wave_travel(depth, wet, mouth_rc, mask == 2, dx_m)


def lag_and_window(
    t_river_hr: float,
    t_surge_hr: float,
    tau95_hr: float,
    tau50_hr: float,
    before_hr: float,
    after_hr: float,
) -> dict:
    """Return the lag, its materiality and the run length needed without and with the shift (t = 0 at the surge peak)."""
    lag = t_river_hr - t_surge_hr
    material = bool(np.isfinite(tau95_hr) and lag > tau95_hr)
    after_noshift = t_river_hr + AFTER_MARGIN_HR
    before_shift = max(lag, 0.0) + tau50_hr
    after_shift = t_surge_hr + AFTER_MARGIN_HR
    ok_noshift = after_hr >= after_noshift
    ok_shift = before_hr >= before_shift and after_hr >= after_shift
    return {
        "t_river_hr": t_river_hr,
        "lag_hr": lag,
        "lag_material": material,
        "after_needed_noshift_hr": after_noshift,
        "before_needed_shift_hr": before_shift,
        "after_needed_shift_hr": after_shift,
        "window_ok": bool(ok_shift if material else ok_noshift),
    }


def _nearest(x: np.ndarray, y: np.ndarray, px: float, py: float) -> int:
    return int(np.argmin(np.hypot(x - px, y - py)))


def estimate_basin(
    results_dir, basin_id: str, scenario: str, manning_n: float, out_dir
) -> pd.DataFrame | None:
    from src.river_network import build_downstream_adjacency, normalize_reach_id
    from src.surge import read_calm_sea_level_m

    base = results_dir / basin_id
    nodes_path = (
        base
        / "preprocessing_inputs"
        / "visuals"
        / "11_calibration"
        / "river_bed"
        / f"{basin_id}_river_bed_nodes.csv"
    )
    run = base / "runs" / scenario / "sfincs"
    gis = base / "sfincs_skeleton" / "gis"
    missing = [
        p
        for p in (nodes_path, run / "sfincs.dis", run / "sfincs.bzs", gis / "dep.tif")
        if not p.exists()
    ]
    if missing:
        log.info(f"{basin_id}: skipped, missing {', '.join(str(p) for p in missing)}")
        return None

    inp = read_inp(run / "sfincs.inp")
    tref, tstart, tstop = (parse_timestamp(inp[k]) for k in ("tref", "tstart", "tstop"))
    offset_hr = (tstart - tref).total_seconds() / 3600.0
    dis = np.loadtxt(run / "sfincs.dis", ndmin=2)
    src = np.loadtxt(run / "sfincs.src", usecols=(0, 1), ndmin=2)
    bzs = np.loadtxt(run / "sfincs.bzs", ndmin=2)
    bnd = np.loadtxt(run / "sfincs.bnd", usecols=(0, 1), ndmin=2)
    main = int(np.argmax(dis[:, 1:].max(axis=0)))
    t_q_hr = dis[:, 0] / 3600.0 - offset_hr
    q_main = dis[:, 1 + main]
    i_surge = int(np.argmax(bzs[:, 1:].max(axis=1)))
    t_surge_peak_hr = bzs[i_surge, 0] / 3600.0 - offset_hr
    before_hr = t_surge_peak_hr
    after_hr = (tstop - tstart).total_seconds() / 3600.0 - t_surge_peak_hr

    nodes = pd.read_csv(nodes_path)
    nodes["reach_id"] = nodes["reach_id"].map(normalize_reach_id)
    rivers = gpd.read_file(
        base
        / "preprocessing_inputs"
        / "domain"
        / f"{basin_id}_river_network_depth_estimated.gpkg"
    )
    downstream = build_downstream_adjacency(rivers)
    sx, sy = src[main]
    reach = nodes["reach_id"].iloc[
        _nearest(nodes["x"].to_numpy(), nodes["y"].to_numpy(), sx, sy)
    ]
    prof = mainstem_profile(nodes, downstream, reach)
    prof = prof.iloc[
        _nearest(prof["x"].to_numpy(), prof["y"].to_numpy(), sx, sy) :
    ].reset_index(drop=True)
    mx, my = prof["x"].iloc[-1], prof["y"].iloc[-1]
    sea_level_m = float(bzs[i_surge, 1 + _nearest(bnd[:, 0], bnd[:, 1], mx, my)])
    with xr.open_dataset(
        base / "preprocessing_inputs" / "forcing" / "surge_forcing.nc",
        decode_times=False,
    ) as ds:
        calm_m = read_calm_sea_level_m(ds)

    q_peak = float(q_main.max())
    leg = river_leg(
        prof, scaled_discharge(prof, q_peak), sea_level_m, calm_m, manning_n
    )

    with (
        rasterio.open(gis / "dep.tif") as r_dep,
        rasterio.open(gis / "mask.tif") as r_msk,
    ):
        dep = r_dep.read(1, masked=True).filled(np.nan).astype(float)
        msk = r_msk.read(1)
        mouth_rc = r_dep.index(mx, my)
        dx_m = abs(r_dep.transform.a)
    t_surge_hr, sea_km = sea_leg(dep, msk, sea_level_m, mouth_rc, dx_m)

    tau95 = hours_before_peak_within(t_q_hr, q_main, 0.95)
    tau50 = hours_before_peak_within(t_q_hr, q_main, 0.50)
    common = {
        "basin_id": basin_id,
        "scenario": scenario,
        "path_km": float(prof["dist_from_mouth_m"].iloc[0]) / 1e3,
        "drowned_share": float(leg["drowned"].mean()),
        "q_peak_m3s": q_peak,
        "q_peak_over_bankfull": q_peak / float(prof["q_design_m3s"].iloc[0]),
        "sea_level_peak_m": sea_level_m,
        "calm_sea_level_m": calm_m,
        "sea_path_km": sea_km,
        "t_surge_hr": t_surge_hr,
        "tau95_hr": tau95,
        "tau50_hr": tau50,
        "before_hr": before_hr,
        "after_hr": after_hr,
    }
    rows = [
        common
        | {"celerity": name}
        | lag_and_window(
            float(leg[f"t_{name}_hr"].iloc[-1]),
            t_surge_hr,
            tau95,
            tau50,
            before_hr,
            after_hr,
        )
        for name in CELERITIES
    ]
    leg.to_csv(
        out_dir / f"{basin_id}_{scenario}_path.csv", index=False, float_format="%.4f"
    )
    _plot(leg, common, out_dir / f"{basin_id}_{scenario}_peak_travel.png")
    return pd.DataFrame(rows)


def _plot(leg: pd.DataFrame, common: dict, path) -> None:
    km = leg["dist_from_mouth_m"] / 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(km, leg["bed_m"], color="0.4", label="bed")
    ax1.plot(
        km,
        leg["level_m"],
        color="tab:blue",
        label=f"level at {common['q_peak_m3s']:.0f} m3/s",
    )
    ax1.axhline(common["calm_sea_level_m"], color="k", ls=":", lw=0.8, label="calm sea")
    ax1.axhline(
        common["sea_level_peak_m"],
        color="tab:red",
        ls="--",
        lw=0.8,
        label="storm-tide peak",
    )
    ax1.set_ylabel("m")
    ax1.legend(fontsize=8)
    for name, color in zip(CELERITIES, ("tab:green", "tab:orange", "tab:purple")):
        ax2.plot(km, leg[f"t_{name}_hr"], color=color, label=name)
    ax2.axhline(
        common["tau95_hr"] + common["t_surge_hr"],
        color="k",
        ls="--",
        lw=0.8,
        label="tau95 + T_surge (lag material above)",
    )
    ax2.set_ylim(0, 72)
    ax2.set_xlabel("distance from mouth (km)")
    ax2.set_ylabel("river peak travel time from forcing point (h)")
    ax2.invert_xaxis()
    ax2.legend(fontsize=8)
    fig.suptitle(
        f"{common['basin_id']} {common['scenario']} peak travel ({_git_hash()}): "
        f"T_surge {common['t_surge_hr']:.1f} h, tau95 {common['tau95_hr']:.1f} h"
    )
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        help="basin ids (default: every basin with a node table)",
    )
    parser.add_argument("--scenario", default="default")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = site_path("results_dir")
    manning_n = float(_config()["river_processing"]["channel_manning_n"])
    out_dir = REPO_ROOT / "figs" / "peak_travel"
    out_dir.mkdir(parents=True, exist_ok=True)
    basin_ids = args.basin_ids or sorted(
        p.name for p in results_dir.iterdir() if p.name.isdigit()
    )
    tables = [
        t
        for b in basin_ids
        if (t := estimate_basin(results_dir, b, args.scenario, manning_n, out_dir))
        is not None
    ]
    if not tables:
        raise SystemExit(
            f"No basin with a node table and a built {args.scenario!r} scenario under {results_dir}"
        )
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out_dir / "peak_travel.csv", index=False, float_format="%.2f")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
