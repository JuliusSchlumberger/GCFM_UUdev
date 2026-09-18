"""
check_crest_calibration.py -- Independent check of rule modelled_depth_estimation's
(10_depth_estimation_modelled.py) weir crests against its own round-2
verification run: does ANY weir segment get overtopped?

Deliberately does not reuse the rule's own bookkeeping (crest_surface /
edge_water_side_mask): every segment of the exported weir gpkg is checked
directly against round 2's SFINCS zsmax (max over every computational
timestep) on BOTH sides of the segment -- SFINCS passes flow over a weir as
soon as either side exceeds the crest.

Checks / reports:
  1. Per-segment margin = crest - max(zsmax left, zsmax right); < 0 means
     overtopped in round 2. Split into river-bank vs coastal segments
     (coastal = the water side is landuse==200 ocean).
  2. Crests SFINCS actually read (round2/sfincs.weir, 0.1 m precision) equal
     the gpkg crests, and all sit on the 0.1 m grid; every gpkg segment
     missing from sfincs.weir lies outside the active domain (no real gap).
  3. Centerline water level round 1 vs round 2 (calibration_state.csv) --
     should be (near-)identical: the finite crests of round 2 hold back the
     same water the 1000 m walls of round 1 did.

Figure (figs/crest_calibration_check/{basin_id}_crest_margin.png): weir
segments coloured by margin (map) + margin histogram per segment type.

Usage:
    conda run -n hmt_sfincs_dev python tests/check_crest_calibration.py [basin_id]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml
from affine import Affine
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.protection_weir import LANDUSE_SEA, OCEAN_WETLAND_CLASSES
from src.sfincs_run import parse_sfincs_inp

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = sys.argv[1] if len(sys.argv) > 1 else "2433835"

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
FIGS_DIR = REPO_ROOT / "figs" / "crest_calibration_check"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

prep = RESULTS_DIR / BASIN_ID / "preprocessing_inputs"
calib_root = prep / "depth_crest_calibration"
round2 = calib_root / "round2"
weir_path = prep / "domain" / f"{BASIN_ID}_coastal_protection_weir.gpkg"
landuse_path = prep / "domain" / f"{BASIN_ID}_landuse_on_grid.tif"

# ── round 2's own grid + zsmax ────────────────────────────────────────────────
inp = parse_sfincs_inp(round2 / "sfincs.inp")
if float(inp.get("rotation", 0.0)) != 0.0:
    raise NotImplementedError("rotated grids are not supported by this check")
dx, dy = float(inp["dx"]), float(inp["dy"])
nmax, mmax = int(inp["nmax"]), int(inp["mmax"])
transform = Affine(dx, 0.0, float(inp["x0"]), 0.0, dy, float(inp["y0"]))
inv = ~transform

with xr.open_dataset(round2 / "sfincs_map.nc") as ds:
    zsmax = ds["zsmax"]
    zsmax = zsmax.max(dim=[d for d in zsmax.dims if d not in ("n", "m")]).values
    map_x, map_y = ds["x"].values, ds["y"].values
# sfincs_map.nc's (n, m) must be this grid's (row, col) -- checked on real
# coordinates rather than assumed.
_rows, _cols = np.indices((nmax, mmax))
_cx, _cy = transform * (_cols + 0.5, _rows + 0.5)
assert (
    map_x.shape == (nmax, mmax)
    and np.allclose(map_x, _cx, atol=1.0)
    and np.allclose(map_y, _cy, atol=1.0)
), "sfincs_map.nc (n, m) does not line up with sfincs.inp's grid"

with rasterio.open(landuse_path) as src:
    landuse = src.read(1)
if landuse.shape != (nmax, mmax):
    raise ValueError(f"landuse_on_grid {landuse.shape} vs model grid {(nmax, mmax)}")
ocean = landuse == LANDUSE_SEA
wetland = np.isin(landuse, OCEAN_WETLAND_CLASSES)

# ── per-segment: water level on both sides ───────────────────────────────────
weir = gpd.read_file(weir_path)
crest = weir["elevation"].to_numpy(float)
n_seg = len(weir)


def _cell(x: float, y: float) -> tuple[int, int] | None:
    c, r = inv * (x, y)
    r, c = int(np.floor(r)), int(np.floor(c))
    return (r, c) if 0 <= r < nmax and 0 <= c < mmax else None


zs_side = np.full((n_seg, 2), np.nan)
is_coastal = np.zeros(n_seg, dtype=bool)
touches_wetland = np.zeros(n_seg, dtype=bool)
for i, geom in enumerate(weir.geometry):
    (x0, y0), (x1, y1) = geom.coords[0][:2], geom.coords[-1][:2]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    horizontal = abs(y1 - y0) < abs(x1 - x0)
    sides = (
        [(mx, my - dy / 2), (mx, my + dy / 2)]
        if horizontal
        else [(mx - dx / 2, my), (mx + dx / 2, my)]
    )
    for k, (sx, sy) in enumerate(sides):
        rc = _cell(sx, sy)
        if rc is not None:
            zs_side[i, k] = zsmax[rc]
            is_coastal[i] |= bool(ocean[rc])
            touches_wetland[i] |= bool(wetland[rc])
# Wetland-edge segments: a wetland/lagoon cell (OCEAN_WETLAND_CLASSES) on one
# side and no open sea -- with unprotected_ocean_wetlands, the dike along the
# landward edge of an ocean-linked wetland (a wetland cell can also sit on the
# protected side, so this is a label, not proof of which side is seaward).
is_wetland_edge = touches_wetland & ~is_coastal
categories = (
    ("river-bank / other", ~is_coastal & ~is_wetland_edge, "steelblue"),
    ("coastal (open sea)", is_coastal, "darkorange"),
    ("wetland edge", is_wetland_edge, "seagreen"),
)

zs_high = np.nanmax(np.where(np.isnan(zs_side), -np.inf, zs_side), axis=1)
zs_high[np.isinf(zs_high)] = np.nan  # both sides dry / outside the grid
margin = crest - zs_high
overtopped = margin < 0

log.info(
    f"Basin {BASIN_ID}: {n_seg} weir segment(s): "
    + ", ".join(f"{int(sel.sum())} {label}" for label, sel, _ in categories)
)
for label, sel, _ in categories:
    m = margin[sel]
    m = m[np.isfinite(m)]
    if len(m):
        log.info(
            f"  {label:18s}: margin crest - max(zsmax both sides): min {m.min():+.3f} m, "
            f"median {np.median(m):+.3f} m, max {m.max():+.3f} m, "
            f"overtopped {int((m < 0).sum())}/{len(m)}"
        )
log.info(f"  segments with both sides dry: {int(np.isnan(margin).sum())}")
if overtopped.any():
    worst = np.argsort(margin)[: min(10, int(overtopped.sum()))]
    log.warning("  OVERTOPPED segments (worst first):")
    for i in worst:
        log.warning(
            f"    #{i}: crest {crest[i]:.2f} m, zsmax sides {zs_side[i]}, margin {margin[i]:+.3f} m"
        )
else:
    log.info("  -> no segment overtopped in round 2")

# ── crests as SFINCS read them (round2/sfincs.weir) ──────────────────────────
# hydromt drops segments outside the model's active region when the weir is
# set, so sfincs.weir can hold fewer segments than the gpkg -- matched by
# midpoint; every dropped one must have INACTIVE cells on both sides (no
# flow is computed there anyway), otherwise the weir has a real gap.
file_mid, file_z = [], []
with open(round2 / "sfincs.weir") as fh:
    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
i = 0
while i < len(lines):
    n_rows = int(lines[i + 1].split()[0])
    pts = np.array(
        [[float(v) for v in ln.split()[:3]] for ln in lines[i + 2 : i + 2 + n_rows]]
    )
    file_mid.append((pts[:-1, :2] + pts[1:, :2]) / 2)
    file_z.append((pts[:-1, 2] + pts[1:, 2]) / 2)
    i += 2 + n_rows
file_mid, file_z = np.concatenate(file_mid), np.concatenate(file_z)
gpkg_mid = np.array(
    [g.interpolate(0.5, normalized=True).coords[0][:2] for g in weir.geometry]
)
dist, idx = cKDTree(file_mid).query(gpkg_mid)
in_file = dist < 1.0
with xr.open_dataset(round2 / "sfincs_map.nc") as ds:
    msk = ds["msk"].values
side_active = np.zeros((n_seg, 2), dtype=bool)
for j, geom in enumerate(weir.geometry):
    (x0, y0), (x1, y1) = geom.coords[0][:2], geom.coords[-1][:2]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    horizontal = abs(y1 - y0) < abs(x1 - x0)
    sides = (
        [(mx, my - dy / 2), (mx, my + dy / 2)]
        if horizontal
        else [(mx - dx / 2, my), (mx + dx / 2, my)]
    )
    for k, (sx, sy) in enumerate(sides):
        rc = _cell(sx, sy)
        side_active[j, k] = rc is not None and msk[rc] > 0
dropped_real_gap = ~in_file & side_active.any(axis=1)
log.info(
    f"  sfincs.weir: {int(in_file.sum())}/{n_seg} gpkg segment(s) present; "
    f"{int((~in_file & ~side_active.any(axis=1)).sum())} dropped outside the active domain (harmless), "
    f"{int(dropped_real_gap.sum())} dropped next to an ACTIVE cell (a real gap)"
)
if dropped_real_gap.any():
    log.warning("  -> the weir SFINCS ran with has gaps inside the active domain")
log.info(
    f"  crest as read by SFINCS vs gpkg: max |diff| {np.max(np.abs(file_z[idx[in_file]] - crest[in_file])):.4f} m; "
    f"gpkg crests on the 0.1 m grid: {np.allclose(crest * 10, np.round(crest * 10))}"
)

# ── centerline water level, round 1 vs round 2 ───────────────────────────────
s1 = pd.read_csv(calib_root / "round1" / "calibration_state.csv")
s2 = pd.read_csv(calib_root / "round2" / "calibration_state.csv")
d = (s2["zs"] - s1["zs"]).to_numpy()
d = d[np.isfinite(d)]
log.info(
    f"  centerline zs round 2 - round 1: min {d.min():+.3f} m, median {np.median(d):+.4f} m, "
    f"max {d.max():+.3f} m ({len(d)} cells)"
)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_map, ax_hist) = plt.subplots(
    1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.4, 1]}
)
segs = [np.asarray(g.coords)[:, :2] for g in weir.geometry]
vlim = 0.5
lc = LineCollection(segs, cmap="RdBu", norm=plt.Normalize(-vlim, vlim), linewidths=1.6)
lc.set_array(np.clip(np.nan_to_num(margin, nan=vlim), -vlim, vlim))
ax_map.add_collection(lc)
ax_map.autoscale()
ax_map.set_aspect("equal")
fig.colorbar(
    lc,
    ax=ax_map,
    label=f"crest - round-2 zsmax (m), clipped at ±{vlim} (red = overtopped)",
)
if overtopped.any():
    ov = np.array(
        [
            g.interpolate(0.5, normalized=True).coords[0][:2]
            for g in weir.geometry[overtopped]
        ]
    )
    ax_map.scatter(
        ov[:, 0],
        ov[:, 1],
        s=60,
        facecolors="none",
        edgecolors="lime",
        linewidths=2,
        label=f"{int(overtopped.sum())} overtopped",
        zorder=5,
    )
    ax_map.legend(loc="upper right")
ax_map.set_title(f"Basin {BASIN_ID}: weir margin per segment, round 2 (verification)")

bins = np.linspace(min(-0.2, np.nanmin(margin)), min(np.nanmax(margin), 3.0), 60)
for label, sel, color in categories:
    m = margin[sel]
    ax_hist.hist(m[np.isfinite(m)], bins=bins, alpha=0.7, color=color, label=label)
ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
ax_hist.set_xlabel("crest - max(zsmax both sides) (m)")
ax_hist.set_ylabel("segments")
ax_hist.set_title("Margin distribution (< 0 = overtopped)")
ax_hist.legend()
fig.tight_layout()
out = FIGS_DIR / f"{BASIN_ID}_crest_margin.png"
fig.savefig(out, dpi=150)
log.info(f"Figure written: {out}")
