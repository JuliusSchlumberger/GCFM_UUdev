"""
test_bank_elevation_check.py — Diagnose river-bank breach risk: is the DEM
bank high enough next to each conditioned river channel, or could water
spill out of the channel where it isn't supposed to?

For each seed reach's downstream network (same seed/BFS traversal as
10_condition_elevation.py), samples the centerline at DEM pixel spacing via
_sample_line_cells, and at every sample point checks two "bank" cells on
EACH side of the channel: the cell just outside the half-width buffer
(distance = width/2 from the centerline) and the next cell out (distance =
width/2 + pixel_size) — a 2-pixel-wide bank strip (60 m on 30 m FathomDEM).
Bank elevations are read from elevation_merged.tif (the raw, per-basin
FathomDEM mosaic) rather than elevation_conditioned.tif, since conditioning
(enforce_river_monotonicity) only ever lowers centerline pixels — bank
pixels are identical in both rasters, so the raw DEM is the natural choice.

Each bank sample is compared against elevation_conditioned.tif's value at
that SAME along-centerline position (the channel invert benchmark). Any
bank cell at or below that benchmark is flagged: physically, that means the
DEM doesn't rise from the channel to the bank at all at that point, so
SFINCS has no bank there to hold water in the channel — flood water could
spill out cross-country instead of following the intended river path.

Layout: one subplot per seed (not 2 columns like rule 10's before/after
plot — the comparison here is bank vs. final channel, not before vs. after
conditioning), with the conditioned channel drawn as a single benchmark
line and both banks scattered around it. Flagged (at/below benchmark) bank
cells are drawn as red squares so they stand out from the normal scatter.

Requires elevation_conditioned.tif to already exist for the basin (rule 10,
enforce_river_monotonicity) — run
`snakemake --config target_basins="[...]" -- <path to *_elevation_conditioned.tif>`
first if it doesn't.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_bank_elevation_check.py            # default basin
    conda run -n hmt_sfincs_dev python tests/test_bank_elevation_check.py 1234567     # a different basin
"""

import logging
import sys
from collections import deque
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.river_network import (
    _as_linestring,
    _sample_line_cells,
    build_downstream_adjacency,
    normalize_reach_id,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = sys.argv[1] if len(sys.argv) > 1 else "4267691"
WIDTH_COLUMN = (
    "width"  # SWORD reach-average channel width -- matches rivwth in 13_build_sfincs.py
)

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
FIGS_DIR = REPO_ROOT / "figs" / "bank_elevation_check"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

domain_dir = RESULTS_DIR / BASIN_ID / "inputs" / "domain"
river_path = domain_dir / f"{BASIN_ID}_river_network_processed.gpkg"
merged_path = domain_dir / f"{BASIN_ID}_elevation_merged.tif"
cond_path = domain_dir / f"{BASIN_ID}_elevation_conditioned.tif"

for p in (river_path, merged_path, cond_path):
    if not p.exists():
        raise FileNotFoundError(
            f"Required input not found: {p} -- run rule enforce_river_monotonicity "
            f'(`snakemake --config target_basins="[{BASIN_ID}]"` targeting '
            f"{cond_path.name}) first."
        )

rivers = gpd.read_file(river_path)
log.info(f"Loaded {len(rivers)} reaches from {river_path.name}")

with rasterio.open(merged_path) as _src:
    merged_arr = _src.read(1).astype(np.float32)
    transform = _src.transform
    merged_nd = _src.nodata
    raster_crs = _src.crs
    pixel_size = abs(_src.transform.a)

with rasterio.open(cond_path) as _src:
    cond_arr = _src.read(1).astype(np.float32)
    cond_nd = _src.nodata

step_m = pixel_size
rivers_proj = rivers.to_crs(raster_crs)

line_by_rid: dict[str, object] = {}
length_by_rid: dict[str, float] = {}
width_by_rid: dict[str, float] = {}
for _row in rivers_proj.itertuples(index=False):
    _rid = normalize_reach_id(_row.reach_id)
    if _rid is None:
        continue
    _g = _as_linestring(_row.geometry)
    if _g is not None and _g.length > 0:
        line_by_rid[_rid] = _g
        length_by_rid[_rid] = _g.length
        _w = getattr(_row, WIDTH_COLUMN, np.nan)
        width_by_rid[_rid] = float(_w) if pd.notna(_w) and _w > 0 else np.nan

downstream_adj = build_downstream_adjacency(rivers)
seeds = [
    normalize_reach_id(r.reach_id)
    for r in rivers.itertuples(index=False)
    if not pd.isna(getattr(r, "is_seed", None)) and bool(getattr(r, "is_seed", False))
]


def _value_at(arr, nd, row, col) -> float:
    v = float(arr[row, col])
    if (nd is not None and v == nd) or not np.isfinite(v):
        return np.nan
    return v


def _normal(line, d: float) -> np.ndarray:
    """Unit normal vector at arc-length d along line (finite-difference tangent, rotated 90°)."""
    eps = min(1.0, line.length / 10.0) if line.length > 0 else 1.0
    d0, d1 = max(0.0, d - eps), min(line.length, d + eps)
    p0, p1 = line.interpolate(d0), line.interpolate(d1)
    tangent = np.array([p1.x - p0.x, p1.y - p0.y])
    n = np.linalg.norm(tangent)
    tangent = tangent / n if n > 0 else np.array([1.0, 0.0])
    return np.array([-tangent[1], tangent[0]])


def _sample_reach(rid: str, dist_from_seed: float) -> dict:
    """
    Sample one reach's centerline (benchmark) and both bank rings.

    Only centerline samples with a finite elevation_conditioned value are
    kept (that value is the benchmark every bank sample on that cross
    -section is compared against). Returns parallel lists:
      x_km, cond_elev              -- benchmark, one entry per centerline sample
      bank_x_km, bank_elev, bank_low  -- one entry per valid bank sample
                                         (both sides, both ring distances)
    """
    line = line_by_rid.get(rid)
    half_w = width_by_rid.get(rid)
    out = {
        "x_km": [],
        "cond_elev": [],
        "bank_x_km": [],
        "bank_elev": [],
        "bank_low": [],
    }
    if line is None or half_w is None or not np.isfinite(half_w):
        return out
    half_w = half_w / 2.0
    ring_dists = (half_w, half_w + step_m)

    for c in _sample_line_cells(line, transform, cond_arr.shape, step_m):
        cond_v = _value_at(cond_arr, cond_nd, c["row"], c["col"])
        if not np.isfinite(cond_v):
            continue
        x_km = (dist_from_seed + c["along_m"]) / 1000.0
        out["x_km"].append(x_km)
        out["cond_elev"].append(cond_v)

        normal = _normal(line, c["along_m"])
        pt = c["point"]
        for side in (+1.0, -1.0):
            for rd in ring_dists:
                ox, oy = pt.x + normal[0] * rd * side, pt.y + normal[1] * rd * side
                row_o, col_o = rasterio.transform.rowcol(transform, ox, oy)
                if not (
                    0 <= row_o < merged_arr.shape[0]
                    and 0 <= col_o < merged_arr.shape[1]
                ):
                    continue
                bank_v = _value_at(merged_arr, merged_nd, row_o, col_o)
                if not np.isfinite(bank_v):
                    continue
                out["bank_x_km"].append(x_km)
                out["bank_elev"].append(bank_v)
                out["bank_low"].append(bool(bank_v <= cond_v))
    return out


# ── per-seed sampling + plot ──────────────────────────────────────────────────

flagged_records: list[dict] = []
n_bank_total = 0
n_low_total = 0

if not seeds:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(
        0.5,
        0.5,
        "No seed reaches found",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="grey",
    )
    seed_summaries = []
else:
    fig, axes = plt.subplots(
        len(seeds), 1, figsize=(12, 4.5 * len(seeds)), squeeze=False
    )
    seed_summaries = []

    for i, seed in enumerate(seeds):
        ax = axes[i][0]

        # BFS downstream from this seed (same traversal as 10_condition_elevation.py)
        dist_from_seed: dict[str, float] = {seed: 0.0}
        queue: deque[str] = deque([seed])
        visit_order: list[str] = [seed]
        while queue:
            rid = queue.popleft()
            for dn in downstream_adj.get(rid, []):
                if dn not in dist_from_seed:
                    dist_from_seed[dn] = dist_from_seed[rid] + length_by_rid.get(
                        rid, 0.0
                    )
                    visit_order.append(dn)
                    queue.append(dn)

        n_bank_seed = 0
        n_low_seed = 0
        for rid in visit_order:
            s = _sample_reach(rid, dist_from_seed[rid])
            if s["x_km"]:
                ax.plot(
                    s["x_km"],
                    s["cond_elev"],
                    color="black",
                    lw=1.2,
                    zorder=3,
                    label="_nolegend_",
                )

            bank_x = np.asarray(s["bank_x_km"])
            bank_e = np.asarray(s["bank_elev"])
            bank_low = np.asarray(s["bank_low"], dtype=bool)
            n_bank_seed += len(bank_x)
            n_low_seed += int(bank_low.sum())

            if bank_low.any():
                ax.scatter(
                    bank_x[bank_low],
                    bank_e[bank_low],
                    marker="s",
                    s=28,
                    color="red",
                    zorder=5,
                    label="_nolegend_",
                )
            if (~bank_low).any():
                ax.scatter(
                    bank_x[~bank_low],
                    bank_e[~bank_low],
                    marker="o",
                    s=6,
                    color="steelblue",
                    alpha=0.5,
                    zorder=2,
                    label="_nolegend_",
                )

            for rec_x, rec_e, rec_low in zip(bank_x, bank_e, bank_low):
                if rec_low:
                    flagged_records.append(
                        {
                            "seed": seed,
                            "reach_id": rid,
                            "x_km": float(rec_x),
                            "bank_elev_m": float(rec_e),
                        }
                    )

        n_bank_total += n_bank_seed
        n_low_total += n_low_seed
        frac = n_low_seed / n_bank_seed if n_bank_seed else 0.0
        seed_summaries.append((seed, n_low_seed, n_bank_seed, frac))

        ax.plot(
            [], [], color="black", lw=1.2, label="elevation_conditioned (benchmark)"
        )
        ax.scatter(
            [], [], marker="o", s=15, color="steelblue", alpha=0.7, label="bank (OK)"
        )
        ax.scatter(
            [], [], marker="s", s=30, color="red", label="bank ≤ benchmark (low)"
        )
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_xlabel("Distance from seed (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(
            f"Seed {seed} — {n_low_seed}/{n_bank_seed} bank cell(s) at/below "
            f"channel invert ({frac:.1%}) — {len(visit_order)} reach(es)"
        )
        ax.grid(True, alpha=0.3)

frac_total = n_low_total / n_bank_total if n_bank_total else 0.0
fig.suptitle(
    f"Basin {BASIN_ID} — bank elevation check ({WIDTH_COLUMN}-based half-width + "
    f"{step_m:.0f} m): {n_low_total}/{n_bank_total} bank cell(s) at/below "
    f"conditioned channel invert ({frac_total:.1%})",
    fontsize=11,
)
fig.tight_layout()
plot_path = FIGS_DIR / f"bank_elevation_check_{BASIN_ID}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Plot written: {plot_path}")

for seed, n_low, n_bank, frac in seed_summaries:
    log.info(f"  seed {seed}: {n_low}/{n_bank} low bank cell(s) ({frac:.1%})")
log.info(
    f"TOTAL: {n_low_total}/{n_bank_total} bank cell(s) at/below the conditioned "
    f"channel invert ({frac_total:.1%}) across {len(seeds)} seed(s)"
)

if flagged_records:
    csv_path = FIGS_DIR / f"bank_elevation_check_{BASIN_ID}_flagged.csv"
    pd.DataFrame(flagged_records).to_csv(csv_path, index=False)
    log.info(f"Flagged bank cells written: {csv_path}")
