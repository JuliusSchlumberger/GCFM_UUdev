"""
test_bank_elevation_check_sfincs.py — Same bank-breach diagnostic as
test_bank_elevation_check.py, but against the ACTUAL built SFINCS grid
(including the subgrid) rather than the raw/conditioned DEM.

test_bank_elevation_check.py checks the pre-model DEM (elevation_merged vs.
elevation_conditioned). This script instead checks the bed level SFINCS will
actually run with: the subgrid reference raster written by rule 13
(quadtree_subgrid.create(write_dep_tif=True) -> dep_subgrid_lev*.tif per
refinement level, or dep_subgrid.tif for a regular grid). That matters
because burn_river_rect (the river-burning step inside subgrid table
creation) can carve/reshape the channel bed differently from what
elevation_conditioned alone would suggest — this test asks whether the
MODEL, as actually built, still has a real bank between channel and land,
not whether the pre-build DEM did.

Both the benchmark (channel centerline) and the bank samples are read from
the SAME subgrid bed-level raster, so this isolates issues introduced by
the build/burn/subgrid process itself, independent of whatever
test_bank_elevation_check.py already found (or didn't) in the raw DEM.

Uses river_network_estuarine.gpkg (rule 13's actual input network — the one
whose 'width' actually got burned into the model), not
river_network_processed.gpkg (used by rule 10's DEM conditioning).

Sampling geometry (centerline pixel-spacing walk, perpendicular bank-ring
offsets at width/2 and width/2 + pixel_size on both sides, seed/downstream
BFS grouping, one subplot per seed, red squares for bank <= benchmark) is
otherwise identical to test_bank_elevation_check.py.

Requires the basin's SFINCS model to already be built (rule 13,
build_sfincs) — run `snakemake --config target_basins="[...]"` targeting
sfincs.inp first if it isn't. No simulation run is required (this checks
the static build, not any simulation output).

Usage:
    conda run -n hmt_sfincs_dev python tests/test_bank_elevation_check_sfincs.py         # default basin
    conda run -n hmt_sfincs_dev python tests/test_bank_elevation_check_sfincs.py 1234567  # a different basin
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
from src.postprocessing import _mosaic_quadtree_dep_levels
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
MOSAIC_MAX_BYTES = 1.5e9  # same order of magnitude as postprocessing.STATS_MAX_BYTES

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
FIGS_DIR = REPO_ROOT / "figs" / "bank_elevation_check_sfincs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

domain_dir = RESULTS_DIR / BASIN_ID / "inputs" / "domain"
sfincs_root = RESULTS_DIR / BASIN_ID / "sfincs"
river_path = domain_dir / f"{BASIN_ID}_river_network_estuarine.gpkg"
sfincs_inp_path = sfincs_root / "sfincs.inp"

if not sfincs_inp_path.exists():
    raise FileNotFoundError(
        f"No built SFINCS model found at {sfincs_inp_path} -- run "
        f'`snakemake --config target_basins="[{BASIN_ID}]"` targeting sfincs.inp first.'
    )
if not river_path.exists():
    raise FileNotFoundError(f"Required input not found: {river_path}")

# ── load the SFINCS subgrid bed level (the SAME raster serves as both the
# centerline benchmark and the bank samples) ──────────────────────────────────
subgrid_dir = sfincs_root / "subgrid"
level_paths = sorted(subgrid_dir.glob("dep_subgrid_lev*.tif"))
dep_subgrid_path = subgrid_dir / "dep_subgrid.tif"

if level_paths:
    log.info(f"Quadtree subgrid: mosaicking {len(level_paths)} refinement level(s)")
    mosaic = _mosaic_quadtree_dep_levels(level_paths, max_bytes=MOSAIC_MAX_BYTES)
    dep_arr = mosaic.values.astype(np.float32)
    transform = mosaic.rio.transform()
    dep_nd = None  # _mosaic_quadtree_dep_levels already converts nodata to NaN
    raster_crs = mosaic.rio.crs
elif dep_subgrid_path.exists() and dep_subgrid_path.stat().st_size > 0:
    log.info("Regular-grid subgrid: reading dep_subgrid.tif directly")
    with rasterio.open(dep_subgrid_path) as _src:
        dep_arr = _src.read(1).astype(np.float32)
        transform = _src.transform
        dep_nd = _src.nodata
        raster_crs = _src.crs
else:
    raise FileNotFoundError(
        f"No subgrid bed-level raster found under {subgrid_dir} "
        f"(dep_subgrid_lev*.tif or dep_subgrid.tif) — was the model built with "
        f"sfincs.subgrid.enabled = true?"
    )

pixel_size = abs(transform.a)
step_m = pixel_size
dep_shape = dep_arr.shape
log.info(f"Bed-level raster: shape={dep_shape}, pixel size={pixel_size:.1f} m")

rivers = gpd.read_file(river_path)
log.info(f"Loaded {len(rivers)} reaches from {river_path.name}")
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
    Sample one reach's centerline (benchmark) and both bank rings, all from
    the SAME subgrid bed-level raster.

    Returns parallel lists:
      x_km, bench_elev               -- benchmark, one entry per centerline sample
      bank_x_km, bank_elev, bank_low  -- one entry per valid bank sample
                                         (both sides, both ring distances)
    """
    line = line_by_rid.get(rid)
    half_w = width_by_rid.get(rid)
    out = {
        "x_km": [],
        "bench_elev": [],
        "bank_x_km": [],
        "bank_elev": [],
        "bank_low": [],
    }
    if line is None or half_w is None or not np.isfinite(half_w):
        return out
    half_w = half_w / 2.0
    ring_dists = (half_w, half_w + step_m)

    for c in _sample_line_cells(line, transform, dep_shape, step_m):
        bench_v = _value_at(dep_arr, dep_nd, c["row"], c["col"])
        if not np.isfinite(bench_v):
            continue
        x_km = (dist_from_seed + c["along_m"]) / 1000.0
        out["x_km"].append(x_km)
        out["bench_elev"].append(bench_v)

        normal = _normal(line, c["along_m"])
        pt = c["point"]
        for side in (+1.0, -1.0):
            for rd in ring_dists:
                ox, oy = pt.x + normal[0] * rd * side, pt.y + normal[1] * rd * side
                row_o, col_o = rasterio.transform.rowcol(transform, ox, oy)
                if not (0 <= row_o < dep_shape[0] and 0 <= col_o < dep_shape[1]):
                    continue
                bank_v = _value_at(dep_arr, dep_nd, row_o, col_o)
                if not np.isfinite(bank_v):
                    continue
                out["bank_x_km"].append(x_km)
                out["bank_elev"].append(bank_v)
                out["bank_low"].append(bool(bank_v <= bench_v))
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
                    s["bench_elev"],
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
            [], [], color="black", lw=1.2, label="SFINCS subgrid bed level (benchmark)"
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
            f"SFINCS channel bed ({frac:.1%}) — {len(visit_order)} reach(es)"
        )
        ax.grid(True, alpha=0.3)

frac_total = n_low_total / n_bank_total if n_bank_total else 0.0
fig.suptitle(
    f"Basin {BASIN_ID} — SFINCS grid bank elevation check ({WIDTH_COLUMN}-based "
    f"half-width + {step_m:.1f} m subgrid pixel): {n_low_total}/{n_bank_total} bank "
    f"cell(s) at/below channel bed ({frac_total:.1%})",
    fontsize=11,
)
fig.tight_layout()
plot_path = FIGS_DIR / f"bank_elevation_check_sfincs_{BASIN_ID}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Plot written: {plot_path}")

for seed, n_low, n_bank, frac in seed_summaries:
    log.info(f"  seed {seed}: {n_low}/{n_bank} low bank cell(s) ({frac:.1%})")
log.info(
    f"TOTAL: {n_low_total}/{n_bank_total} bank cell(s) at/below the SFINCS channel "
    f"bed ({frac_total:.1%}) across {len(seeds)} seed(s)"
)

if flagged_records:
    csv_path = FIGS_DIR / f"bank_elevation_check_sfincs_{BASIN_ID}_flagged.csv"
    pd.DataFrame(flagged_records).to_csv(csv_path, index=False)
    log.info(f"Flagged bank cells written: {csv_path}")
