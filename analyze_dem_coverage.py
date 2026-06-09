"""
analyze_dem_coverage.py — Compare spatial coverage of DiluviumDEM vs DeltaDTM
over delta polygons, masking offshore areas with OSM land polygons.

Coverage is defined as:
    valid DEM land pixels / total OSM land pixels  (within the delta polygon)

Two further analyses use per-basin Snakemake pipeline outputs (only available
for basins where the relevant rule has already run — see `RESULTS_DIR`):

  - River boundary forcing crossings (`results/{basin_id}/inputs/forcing/
    river_forcing.nc`, via `load_forcing_crossings`): for each DEM, what
    fraction of the active GloFAS-matched crossing locations fall on a valid
    (covered) DEM pixel.
  - River-network buffer coverage (`results/{basin_id}/inputs/
    river_network_processed.gpkg`): coverage (valid DEM land pixels / OSM land
    pixels, same definition as above) restricted to 5 / 10 / 30 km buffers
    around the processed river network — the area most relevant to the
    hydrodynamic model.

Paths are read from config/data_catalogue.yml (meta.root + file_path) and
config/config.yml (results_dir). Tile-finding logic is imported directly from
workflow/src/raster.py, crossing loading from workflow/src/river_forcing.py.

Outputs
-------
  figs/dem_coverage/coverage_histogram.png   — per-DEM coverage distributions across all deltas
  figs/dem_coverage/buffer_coverage.png      — per-DEM coverage vs. river-network buffer distance
  figs/dem_coverage/coverage_table.csv       — per-delta coverage statistics

Usage
-----
  python analyze_dem_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import yaml
from rasterio.features import rasterize as rio_rasterize
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine, rowcol
from shapely.geometry.base import BaseGeometry

# ── Import tile finders from the pipeline source ───────────────────────────────
_WORKFLOW_DIR = Path(__file__).parent / "workflow"
if str(_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW_DIR))

from src.raster import find_diluviumdem_tiles, find_deltadtm_tiles  # type: ignore[import]  # noqa: E402
from src.river_forcing import load_forcing_crossings  # type: ignore[import]  # noqa: E402

# ── Paths from data_catalogue.yml / config.yml ────────────────────────────────

_CATALOGUE = Path(__file__).parent / "config" / "data_catalogue.yml"
_CONFIG = Path(__file__).parent / "config" / "config.yml"


def _load_catalogue_paths() -> dict[str, Path]:
    with open(_CATALOGUE) as f:
        cat = yaml.safe_load(f)
    root = Path(cat["meta"]["root"])
    ds = {d["name"]: d for d in cat["datasets"]}
    return {
        "delta_polygons": root / ds["delta_polygons"]["file_path"],
        "diluvium_dem": root / ds["diluvium_dem"]["file_path"],
        "delta_dtm": root / ds["delta_dtm"]["file_path"],
        "osm_land": root / ds["osm_land"]["file_path"],
    }


def _load_results_dir() -> Path:
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["results_dir"])


PATHS = _load_catalogue_paths()
RESULTS_DIR = _load_results_dir()

BASIN_ID_COL = "BasinID2"

OUTPUT_DIR = Path("figs/dem_coverage")
OUTPUT_HIST = OUTPUT_DIR / "coverage_histogram.png"
OUTPUT_BUFFER = OUTPUT_DIR / "buffer_coverage.png"
OUTPUT_TABLE = OUTPUT_DIR / "coverage_table.csv"
COVERAGE_MAP_DIR = OUTPUT_DIR / "per_delta"

# Deltas where either DEM has coverage below this threshold get a spatial map.
COVERAGE_MAP_THRESHOLD = 95.0

# Maximum pixels per dimension when loading DEMs for the coverage map plot.
# The native resolution (≤ 30 m for DiluviumDEM, ≤ 10 m for DeltaDTM) can
# produce arrays of tens-of-thousands of pixels for large deltas; for a
# 14"-wide figure that's unnecessary and risks out-of-memory errors.
COVERAGE_MAP_MAX_PIXELS = 2000

# Buffer distances (km) around the processed river network for the
# river-corridor coverage analysis.
BUFFER_DISTANCES_KM: list[int] = [5, 10, 30]

DEM_OPTIONS: dict[str, tuple] = {
    # (tile_finder, tiles_dir, max_valid_elevation_m)
    # Pixels above the elevation cap are treated as missing coverage: DiluviumDEM
    # is designed for low-lying coastal zones (≤ 30 m) and DeltaDTM for river
    # deltas (≤ 10 m); values above these thresholds are extrapolations or
    # outright undefined, so they should not count as "covered".
    "DiluviumDEM": (find_diluviumdem_tiles, PATHS["diluvium_dem"], 30.0),
    "DeltaDTM": (find_deltadtm_tiles, PATHS["delta_dtm"], 10.0),
}


# ── Per-basin pipeline outputs (results/{basin_id}/...) ───────────────────────
# Only available for basins where the corresponding Snakemake rule has already
# run; loaders below return None when the file is missing so the rest of the
# script degrades gracefully (these analyses are simply skipped for that delta).


def _crossings_path(basin_id: int) -> Path:
    return RESULTS_DIR / str(basin_id) / "inputs" / "forcing" / "river_forcing.nc"


def _river_network_path(basin_id: int) -> Path:
    return RESULTS_DIR / str(basin_id) / "inputs" / "river_network_processed.gpkg"


def load_crossings(basin_id: int) -> gpd.GeoDataFrame | None:
    """Active (GloFAS-matched) river boundary forcing crossings, or None if unavailable."""
    path = _crossings_path(basin_id)
    if not path.exists():
        return None
    try:
        gdf = load_forcing_crossings(path)
    except Exception as exc:
        print(f"  Could not read crossings from {path}: {exc}")
        return None
    return gdf if not gdf.empty else None


def load_river_network(basin_id: int) -> gpd.GeoDataFrame | None:
    """Processed (clipped, cleaned) river network for this basin, or None if unavailable."""
    path = _river_network_path(basin_id)
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        print(f"  Could not read river network from {path}: {exc}")
        return None
    return gdf if not gdf.empty else None


def make_buffer_polygons(
    river_gdf: gpd.GeoDataFrame,
    buffer_km_list: list[int],
) -> dict[int, BaseGeometry]:
    """
    Buffer the dissolved river network at each distance in *buffer_km_list*
    (km). Buffering is done in a local UTM CRS (estimated from the network's
    extent) so distances are metric, then reprojected back to EPSG:4326 to
    match the DEM/land-mask grids.
    """
    utm_crs = river_gdf.estimate_utm_crs()
    river_geom = river_gdf.to_crs(utm_crs).geometry.union_all()
    polygons: dict[int, BaseGeometry] = {}
    for km in buffer_km_list:
        reprojected = gpd.GeoSeries(
            [river_geom.buffer(km * 1_000.0)], crs=utm_crs
        ).to_crs("EPSG:4326")
        polygons[km] = reprojected.iloc[0]  # type: ignore[assignment]
    return polygons


# ── Raster tile load + clip ────────────────────────────────────────────────────


def load_dem(
    tiles: list[str],
    bounds: tuple[float, float, float, float],
    target_res: float | None = None,
) -> tuple[np.ndarray, Affine] | tuple[None, None]:
    """
    Merge *tiles* (already selected for this delta polygon), clip to *bounds*.
    Tiles are opened, merged, then closed — only the pixels for this polygon
    are read, so memory use scales with polygon size, not the full dataset.

    ``target_res`` (CRS units, typically degrees for WGS84 DEMs) sets the
    output pixel size via ``rio_merge``'s ``res`` argument.  Pass a coarser
    value than the native resolution to downsample — useful for visualisation
    when the native resolution would produce arrays too large to allocate.

    Returns (float32 array, transform) or (None, None) if tiles is empty.
    """
    if not tiles:
        return None, None

    open_ds = [rasterio.open(p) for p in tiles]
    try:
        # Read nodata before the merge so it is available after files are closed.
        nodata_in = open_ds[0].nodata
        # nodata=np.nan tells rasterio to output NaN for masked/nodata pixels.
        merge_kwargs: dict = dict(bounds=bounds, nodata=np.nan)
        if target_res is not None:
            merge_kwargs["res"] = target_res
        merged, transform = rio_merge(open_ds, **merge_kwargs)
    finally:
        for ds in open_ds:
            ds.close()

    arr = merged[0].astype(np.float32)
    # Belt-and-suspenders: replace any residual nodata value that rio_merge
    # may not have caught (e.g. DeltaDTM's large-negative sentinel).
    if nodata_in is not None:
        try:
            nd32 = np.float32(nodata_in)
            if np.isfinite(nd32):
                arr[arr == nd32] = np.nan
        except (TypeError, OverflowError):
            pass

    return arr, transform


# ── Land mask ─────────────────────────────────────────────────────────────────


def make_land_mask(
    land_gdf: gpd.GeoDataFrame,
    delta_geom,
    shape: tuple[int, int],
    transform: Affine,
) -> np.ndarray:
    """
    Rasterise the intersection of *land_gdf* with *delta_geom* onto a grid
    defined by *shape* and *transform*.  Returns a boolean array.
    """
    candidates = land_gdf[land_gdf.geometry.intersects(delta_geom)]
    if candidates.empty:
        return np.zeros(shape, dtype=bool)

    clipped = candidates.copy()
    clipped["geometry"] = clipped.geometry.intersection(delta_geom)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    if clipped.empty:
        return np.zeros(shape, dtype=bool)

    return rio_rasterize(
        [(geom, 1) for geom in clipped.geometry if geom is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)


def make_buffer_mask(
    buffer_geom,
    delta_geom,
    land_mask: np.ndarray,
    shape: tuple[int, int],
    transform: Affine,
) -> np.ndarray:
    """
    Rasterise *buffer_geom* (clipped to *delta_geom*, so it stays within the
    area the DEM/land mask were computed over) onto the same grid as
    *land_mask*, and intersect with it — i.e. the land pixels that fall within
    the river-network buffer.
    """
    clipped = buffer_geom.intersection(delta_geom)
    if clipped.is_empty:
        return np.zeros(shape, dtype=bool)

    buffer_mask = rio_rasterize(
        [(clipped, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)
    return buffer_mask & land_mask


# ── Crossing-point coverage ────────────────────────────────────────────────────


def count_points_on_valid_pixels(
    points_gdf: gpd.GeoDataFrame,
    arr: np.ndarray,
    transform: Affine,
    max_elev: float = np.inf,
) -> int:
    """Count how many *points_gdf* points fall on a valid (finite and ≤ max_elev) DEM pixel."""
    if points_gdf.empty:
        return 0

    nrows, ncols = arr.shape
    xs = points_gdf.geometry.x.to_numpy()
    ys = points_gdf.geometry.y.to_numpy()
    rows, cols = rowcol(transform, xs, ys)

    n_covered = 0
    for row, col in zip(rows, cols):
        if (
            0 <= row < nrows
            and 0 <= col < ncols
            and np.isfinite(arr[row, col])
            and arr[row, col] <= max_elev
        ):
            n_covered += 1
    return n_covered


# ── Per-delta analysis ─────────────────────────────────────────────────────────


def analyse_delta(
    basin_id: int,
    delta_geom,
    land_gdf: gpd.GeoDataFrame,
    crossings_gdf: gpd.GeoDataFrame | None = None,
    buffer_polygons: dict[int, BaseGeometry] | None = None,
) -> dict:
    bounds = delta_geom.bounds  # (minx, miny, maxx, maxy)
    row: dict = {"basin_id": basin_id}

    if crossings_gdf is not None:
        row["n_river_crossings"] = len(crossings_gdf)

    # Buffers clipped to the delta polygon once — independent of DEM grid.
    clipped_buffers = (
        {km: geom.intersection(delta_geom) for km, geom in buffer_polygons.items()}
        if buffer_polygons is not None
        else None
    )

    for dem_name, (tile_finder, topo_dir, elev_max) in DEM_OPTIONS.items():
        # Find and load only tiles that intersect this specific polygon's bounds.
        tiles = tile_finder(str(topo_dir), bounds)
        row[f"{dem_name}_n_tiles"] = len(tiles)

        arr = transform = None
        if tiles:
            arr, transform = load_dem(tiles, bounds)

        if arr is None or transform is None:
            row[f"{dem_name}_coverage_pct"] = np.nan
            row[f"{dem_name}_n_land_px"] = np.nan
            row[f"{dem_name}_n_covered_px"] = np.nan
            if crossings_gdf is not None:
                row[f"{dem_name}_crossings_covered_pct"] = np.nan
            if clipped_buffers is not None:
                for km in clipped_buffers:
                    row[f"{dem_name}_buffer_{km}km_coverage_pct"] = np.nan
            continue

        land_mask = make_land_mask(land_gdf, delta_geom, arr.shape, transform)
        n_land = int(land_mask.sum())
        # A pixel counts as "covered" only if it has a finite value AND is at or
        # below the DEM's valid elevation ceiling (DiluviumDEM ≤ 30 m, DeltaDTM ≤ 10 m).
        valid = np.isfinite(arr) & (arr <= elev_max)
        n_covered = int((valid & land_mask).sum())

        row[f"{dem_name}_n_land_px"] = n_land
        row[f"{dem_name}_n_covered_px"] = n_covered
        row[f"{dem_name}_coverage_pct"] = (
            100.0 * n_covered / n_land if n_land > 0 else np.nan
        )

        # (a) Do the river boundary forcing crossings fall on covered pixels?
        if crossings_gdf is not None:
            n_cross = len(crossings_gdf)
            n_cross_covered = count_points_on_valid_pixels(
                crossings_gdf, arr, transform, max_elev=elev_max
            )
            row[f"{dem_name}_crossings_covered_pct"] = (
                100.0 * n_cross_covered / n_cross if n_cross > 0 else np.nan
            )

        # (b) Coverage within buffers around the river network.
        if clipped_buffers is not None:
            for km, buf_geom in clipped_buffers.items():
                buf_mask = make_buffer_mask(
                    buf_geom, delta_geom, land_mask, arr.shape, transform
                )
                n_buf_land = int(buf_mask.sum())
                n_buf_cov = int((valid & buf_mask).sum())
                row[f"{dem_name}_buffer_{km}km_coverage_pct"] = (
                    100.0 * n_buf_cov / n_buf_land if n_buf_land > 0 else np.nan
                )

    return row


# ── Per-delta spatial coverage map ────────────────────────────────────────────


def plot_coverage_map(
    basin_id: int,
    delta_geom,
    land_gdf: gpd.GeoDataFrame,
    output_path: Path,
    crossings_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """
    Two-subplot figure showing which land pixels each DEM covers inside the delta.

    Left  = DiluviumDEM, Right = DeltaDTM.  Valid pixels are coloured by
    elevation (terrain colourmap, 0–20 m); missing land pixels are red; ocean
    / outside-delta pixels are light grey.

    When *crossings_gdf* is given, river boundary forcing crossing locations
    are overlaid — green where they fall on a covered DEM pixel, red where
    they fall on a missing/offshore one — answering "are the model's inflow
    points inside the DEM-covered area?" at a glance.
    """
    bounds = delta_geom.bounds

    # Downsample to at most COVERAGE_MAP_MAX_PIXELS per side so large deltas
    # with high-resolution DEMs (DeltaDTM ≤ 10 m) don't exhaust memory.
    lon_span = bounds[2] - bounds[0]
    lat_span = bounds[3] - bounds[1]
    plot_res = max(lon_span, lat_span) / COVERAGE_MAP_MAX_PIXELS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap_missing = matplotlib.colors.ListedColormap(["#d73027"])

    for ax, (dem_name, (tile_finder, topo_dir, elev_max)) in zip(
        axes, DEM_OPTIONS.items()
    ):
        tiles = tile_finder(str(topo_dir), bounds)
        if not tiles:
            ax.set_title(f"{dem_name}\n(no tiles)")
            ax.axis("off")
            continue

        arr, transform = load_dem(tiles, bounds, target_res=plot_res)
        if arr is None or transform is None:
            ax.set_title(f"{dem_name}\n(load failed)")
            ax.axis("off")
            continue

        nrows, ncols = arr.shape
        left = transform.c
        top = transform.f
        right = left + transform.a * ncols
        bottom = top + transform.e * nrows
        extent = [left, right, bottom, top]

        land_mask = make_land_mask(land_gdf, delta_geom, arr.shape, transform)
        valid = np.isfinite(arr) & (arr <= elev_max)
        valid_land = valid & land_mask
        missing_land = ~valid & land_mask  # no data OR above elevation cap

        n_land = int(land_mask.sum())
        n_covered = int(valid_land.sum())
        pct = 100.0 * n_covered / n_land if n_land > 0 else float("nan")

        # Ocean / outside-delta backdrop (light grey)
        ocean = np.where(~land_mask, 1.0, np.nan)
        ax.imshow(
            ocean,
            extent=extent,
            cmap="Greys",
            vmin=0,
            vmax=2,
            origin="upper",
            interpolation="nearest",
        )

        # Valid DEM elevation
        dem_show = np.where(valid_land, arr, np.nan)
        im = ax.imshow(
            dem_show,
            extent=extent,
            cmap="terrain",
            vmin=0,
            vmax=elev_max,
            origin="upper",
            interpolation="nearest",
        )

        # Missing land pixels
        missing_show = np.where(missing_land, 1.0, np.nan)
        ax.imshow(
            missing_show,
            extent=extent,
            cmap=cmap_missing,
            vmin=0.5,
            vmax=1.5,
            origin="upper",
            interpolation="nearest",
        )

        # Delta polygon boundary (works for both Polygon and MultiPolygon)
        gpd.GeoSeries([delta_geom]).boundary.plot(
            ax=ax, color="black", linewidth=1.2, zorder=5
        )

        # River boundary forcing crossings — green if on a covered DEM pixel,
        # red (matching the "missing land coverage" colour) if not.
        cross_str = ""
        if crossings_gdf is not None and not crossings_gdf.empty:
            xs = crossings_gdf.geometry.x.to_numpy()
            ys = crossings_gdf.geometry.y.to_numpy()
            rows_idx, cols_idx = rowcol(transform, xs, ys)
            on_grid = (
                (np.asarray(rows_idx) >= 0)
                & (np.asarray(rows_idx) < nrows)
                & (np.asarray(cols_idx) >= 0)
                & (np.asarray(cols_idx) < ncols)
            )
            covered = np.zeros(len(xs), dtype=bool)
            r_ok = np.asarray(rows_idx)[on_grid]
            c_ok = np.asarray(cols_idx)[on_grid]
            covered[on_grid] = np.isfinite(arr[r_ok, c_ok]) & (
                arr[r_ok, c_ok] <= elev_max
            )

            ax.scatter(
                xs[covered],
                ys[covered],
                marker="o",
                s=45,
                color="#1a9850",
                edgecolors="black",
                linewidth=0.6,
                zorder=6,
                label="Crossing — covered",
            )
            ax.scatter(
                xs[~covered],
                ys[~covered],
                marker="X",
                s=55,
                color="#d73027",
                edgecolors="black",
                linewidth=0.6,
                zorder=6,
                label="Crossing — not covered",
            )
            cross_str = f"\ncrossings covered: {int(covered.sum())}/{len(xs)}"

        pct_str = f"{pct:.1f}%" if np.isfinite(pct) else "n/a"
        ax.set_title(
            f"{dem_name}  (valid ≤ {elev_max:.0f} m)  —  coverage: {pct_str}\n"
            f"({n_covered:,} / {n_land:,} land pixels){cross_str}",
            fontsize=10,
        )
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.75, pad=0.02)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(color="lightgrey", label="Ocean / outside delta"),
        Patch(color="#d73027", label="Missing land coverage"),
    ]
    ncol = 2
    if crossings_gdf is not None and not crossings_gdf.empty:
        legend_elements += [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor="#1a9850",
                markeredgecolor="black",
                label="Crossing — covered",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="",
                markersize=8,
                markerfacecolor="#d73027",
                markeredgecolor="black",
                label="Crossing — not covered",
            ),
        ]
        ncol = 4
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=ncol,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(f"DEM spatial coverage — basin {basin_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Coverage map: {output_path}")


# ── Histogram plot ─────────────────────────────────────────────────────────────


def plot_histogram(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    bins = np.linspace(0, 100, 21)
    colors = {"DiluviumDEM": "#2166ac", "DeltaDTM": "#d6604d"}

    # Left: overlaid step-line (outline-only) histograms — avoids the
    # occlusion of filled overlapping bars while still comparing distributions.
    ax = axes[0]
    for dem, color in colors.items():
        col = f"{dem}_coverage_pct"
        vals = df[col].dropna().values
        n_ok = len(vals)
        n_na = df[col].isna().sum()
        lbl = f"{dem}  (n={n_ok}"
        if n_na:
            lbl += f", {n_na} missing"
        lbl += ")"
        ax.hist(vals, bins=bins, histtype="step", color=color, label=lbl, linewidth=2.0)

    ax.set_xlabel("Coverage of land area within delta polygon (%)", fontsize=10)
    ax.set_ylabel("Number of delta polygons", fontsize=10)
    ax.set_title("Distribution of DEM coverage", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Right: side-by-side box-plot
    ax2 = axes[1]
    plot_data = [df[f"{dem}_coverage_pct"].dropna().values for dem in colors]
    bp = ax2.boxplot(
        plot_data,
        labels=list(colors.keys()),
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(plot_data, colors.values()), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax2.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color,
            alpha=0.7,
            s=30,
            zorder=3,
            edgecolors="white",
            linewidth=0.3,
        )

    ax2.set_ylabel("Coverage (%)", fontsize=10)
    ax2.set_title("Coverage distribution per DEM", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    fig.suptitle(
        "Spatial coverage of DEM options over delta land areas\n"
        "(valid pixels / OSM land pixels within delta polygon)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written: {output_path}")


# ── River-network buffer coverage plot ────────────────────────────────────────


def plot_buffer_coverage(df: pd.DataFrame, output_path: Path) -> None:
    """
    Coverage (%) within buffers of increasing distance around each delta's
    processed river network, per DEM — shows whether coverage degrades away
    from the river corridor that matters most to the hydrodynamic model.

    Each delta is drawn as a jittered scatter point per buffer distance (the
    same style as the box-plot subplot in `plot_histogram`); per-DEM medians
    are connected by a line to show the overall trend across distances. Only
    deltas with a processed river network (`river_network_processed.gpkg`)
    contribute — typically a subset of all deltas.
    """
    colors = {"DiluviumDEM": "#2166ac", "DeltaDTM": "#d6604d"}
    buffer_cols = {
        dem: [f"{dem}_buffer_{km}km_coverage_pct" for km in BUFFER_DISTANCES_KM]
        for dem in colors
    }
    if not all(col in df.columns for cols in buffer_cols.values() for col in cols):
        print(
            "Skipping buffer-coverage plot — no river-network buffer data available "
            "(river_network_processed.gpkg not found for any delta)"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(42)
    x = np.arange(len(BUFFER_DISTANCES_KM), dtype=float)

    n_deltas = 0
    for i, (dem, color) in enumerate(colors.items()):
        offset = -0.12 if i == 0 else 0.12
        medians = []
        for j, col in enumerate(buffer_cols[dem]):
            vals = np.asarray(df[col].dropna().values, dtype=float)
            n_deltas = max(n_deltas, len(vals))
            medians.append(np.median(vals) if len(vals) else np.nan)
            jitter = rng.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(
                x[j] + offset + jitter,
                vals,
                color=color,
                alpha=0.6,
                s=24,
                edgecolors="white",
                linewidth=0.3,
                zorder=3,
            )
        ax.plot(
            x + offset,
            medians,
            color=color,
            marker="D",
            markersize=7,
            linewidth=2.0,
            label=f"{dem} (median)",
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{km} km" for km in BUFFER_DISTANCES_KM])
    ax.set_xlabel("Buffer distance around processed river network", fontsize=10)
    ax.set_ylabel("Coverage (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_title(
        f"DEM coverage within river-network buffers  (n={n_deltas} deltas with processed network)\n"
        "(valid DEM land pixels / OSM land pixels within buffer)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written: {output_path}")


# ── Summary table ──────────────────────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    print("\n── Per-delta coverage table ──────────────────────────────────────")
    base_cov_cols = [f"{dem}_coverage_pct" for dem in DEM_OPTIONS]
    cols = ["basin_id"] + [
        c for c in df.columns if c in base_cov_cols or "n_tiles" in c
    ]
    out = df[cols].copy()
    for c in out.columns:
        if "coverage_pct" in c:
            out[c] = out[c].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    print(out.to_string(index=False))

    print("\n── Summary statistics ────────────────────────────────────────────")
    for dem in DEM_OPTIONS:
        col = f"{dem}_coverage_pct"
        vals = df[col].dropna()
        print(f"  {dem}:")
        print(f"    Deltas with data     : {len(vals)} / {len(df)}")
        if len(vals) > 0:
            print(f"    Mean coverage        : {vals.mean():.1f}%")
            print(f"    Median coverage      : {vals.median():.1f}%")
            print(f"    Min / Max            : {vals.min():.1f}% / {vals.max():.1f}%")
            print(f"    Fully covered (≥99%) : {(vals >= 99).sum()}")
            print(f"    Partial (<50%)       : {(vals < 50).sum()}")

    # ── (a) river boundary forcing crossings vs. DEM coverage ─────────────────
    if "n_river_crossings" in df.columns:
        has_crossings = df["n_river_crossings"].notna()
        print(
            f"\n── River boundary forcing crossings vs. DEM coverage "
            f"({int(has_crossings.sum())}/{len(df)} deltas with river_forcing.nc) ──"
        )
        for dem in DEM_OPTIONS:
            col = f"{dem}_crossings_covered_pct"
            vals = df.loc[has_crossings, col].dropna()
            print(f"  {dem}:")
            print(
                f"    Deltas with data        : {len(vals)} / {int(has_crossings.sum())}"
            )
            if len(vals) > 0:
                fully = int((vals >= 100.0 - 1e-9).sum())
                partial = int((vals < 100.0 - 1e-9).sum())
                print(f"    Mean crossings covered  : {vals.mean():.1f}%")
                print(f"    All crossings covered   : {fully} delta(s)")
                print(f"    ≥1 crossing not covered : {partial} delta(s)")

    # ── (b) river-network buffer coverage ─────────────────────────────────────
    buffer_cols_present = [
        f"{dem}_buffer_{km}km_coverage_pct"
        for dem in DEM_OPTIONS
        for km in BUFFER_DISTANCES_KM
        if f"{dem}_buffer_{km}km_coverage_pct" in df.columns
    ]
    if buffer_cols_present:
        n_with_network = (
            df[f"DiluviumDEM_buffer_{BUFFER_DISTANCES_KM[0]}km_coverage_pct"]
            .notna()
            .sum()
            if f"DiluviumDEM_buffer_{BUFFER_DISTANCES_KM[0]}km_coverage_pct"
            in df.columns
            else 0
        )
        print(
            f"\n── DEM coverage within river-network buffers "
            f"({int(n_with_network)}/{len(df)} deltas with river_network_processed.gpkg) ──"
        )
        for dem in DEM_OPTIONS:
            print(f"  {dem}:")
            for km in BUFFER_DISTANCES_KM:
                col = f"{dem}_buffer_{km}km_coverage_pct"
                if col not in df.columns:
                    continue
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(
                        f"    {km:>2d} km buffer — mean: {vals.mean():5.1f}%   "
                        f"median: {vals.median():5.1f}%   min: {vals.min():5.1f}%   (n={len(vals)})"
                    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data catalogue: {_CATALOGUE}")
    print(f"  DiluviumDEM tiles : {PATHS['diluvium_dem']}")
    print(f"  DeltaDTM tiles    : {PATHS['delta_dtm']}")

    # Load delta polygons
    print(f"\nLoading delta polygons: {PATHS['delta_polygons']}")
    deltas = gpd.read_file(PATHS["delta_polygons"]).to_crs("EPSG:4326")
    print(f"  {len(deltas)} polygon(s), columns: {list(deltas.columns)}")

    # Pre-load OSM land clipped to the union of all delta bboxes.
    # OSM is a vector dataset; loading once for the total extent avoids
    # repeated file reads while keeping memory reasonable.
    total_bounds = tuple(float(v) for v in deltas.total_bounds)
    print(f"\nLoading OSM land polygons (bbox={total_bounds})…")
    land_gdf = gpd.read_file(
        PATHS["osm_land"], bbox=total_bounds, engine="pyogrio", layer="land_polygons"
    ).to_crs("EPSG:4326")
    print(f"  {len(land_gdf)} land polygon(s)")

    print(f"\nResults dir (per-basin pipeline outputs): {RESULTS_DIR}")
    print(
        "  river boundary forcing crossings : {basin_id}/inputs/forcing/river_forcing.nc"
    )
    print(
        "  processed river network          : {basin_id}/inputs/river_network_processed.gpkg"
    )
    print(f"  buffer distances                 : {BUFFER_DISTANCES_KM} km")

    # Analyse each delta — DEM tiles are found and loaded per polygon,
    # so only the small subset covering each polygon's bounds is read.
    print("\nAnalysing coverage per delta…")
    rows = []
    for _, delta_row in deltas.iterrows():
        basin_id = delta_row.get(BASIN_ID_COL, delta_row.name)
        basin_id = int(basin_id)
        geom = delta_row.geometry

        # Per-basin pipeline outputs — only available once the relevant
        # Snakemake rule has run for this basin; loaders return None otherwise.
        crossings_gdf = load_crossings(basin_id)
        river_gdf = load_river_network(basin_id)
        buffer_polygons = (
            make_buffer_polygons(river_gdf, BUFFER_DISTANCES_KM)
            if river_gdf is not None
            else None
        )

        extra = []
        if crossings_gdf is not None:
            extra.append(f"{len(crossings_gdf)} crossings")
        if buffer_polygons is not None:
            extra.append(f"network buffers {BUFFER_DISTANCES_KM}km")
        extra_str = f"  [{', '.join(extra)}]" if extra else ""

        print(
            f"  [{basin_id}] bounds={tuple(round(v, 3) for v in geom.bounds)}{extra_str}",
            end=" → ",
        )
        try:
            result = analyse_delta(
                basin_id, geom, land_gdf, crossings_gdf, buffer_polygons
            )
            dil = result.get("DiluviumDEM_coverage_pct")
            ddm = result.get("DeltaDTM_coverage_pct")
            dil_str = f"{dil:.1f}%" if dil is not None and np.isfinite(dil) else "n/a"
            ddm_str = f"{ddm:.1f}%" if ddm is not None and np.isfinite(ddm) else "n/a"
            dil_t = result.get("DiluviumDEM_n_tiles", 0)
            ddm_t = result.get("DeltaDTM_n_tiles", 0)
            print(
                f"DiluviumDEM={dil_str} ({dil_t} tiles)  DeltaDTM={ddm_str} ({ddm_t} tiles)"
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {
                "basin_id": basin_id,
                "DiluviumDEM_n_tiles": 0,
                "DiluviumDEM_n_land_px": np.nan,
                "DiluviumDEM_n_covered_px": np.nan,
                "DiluviumDEM_coverage_pct": np.nan,
                "DeltaDTM_n_tiles": 0,
                "DeltaDTM_n_land_px": np.nan,
                "DeltaDTM_n_covered_px": np.nan,
                "DeltaDTM_coverage_pct": np.nan,
            }
        rows.append(result)

        # Create a spatial coverage map when either DEM is below threshold, or
        # when a river boundary forcing crossing falls outside DEM coverage —
        # both are situations worth a closer visual look.
        dil_pct = result.get("DiluviumDEM_coverage_pct")
        ddm_pct = result.get("DeltaDTM_coverage_pct")
        below = (
            dil_pct is not None
            and np.isfinite(dil_pct)
            and dil_pct < COVERAGE_MAP_THRESHOLD
        ) or (
            ddm_pct is not None
            and np.isfinite(ddm_pct)
            and ddm_pct < COVERAGE_MAP_THRESHOLD
        )
        crossing_issue = any(
            (pct := result.get(f"{dem}_crossings_covered_pct")) is not None
            and np.isfinite(pct)
            and pct < 100.0 - 1e-9
            for dem in DEM_OPTIONS
        )
        if below or crossing_issue:
            map_path = COVERAGE_MAP_DIR / f"{basin_id}_coverage_map.png"
            try:
                plot_coverage_map(
                    basin_id, geom, land_gdf, map_path, crossings_gdf=crossings_gdf
                )
            except Exception as exc:
                print(f"  Coverage map failed: {exc}")

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_TABLE, index=False, float_format="%.2f")
    print(f"\nTable written: {OUTPUT_TABLE}")

    print_summary(df)
    plot_histogram(df, OUTPUT_HIST)
    plot_buffer_coverage(df, OUTPUT_BUFFER)


if __name__ == "__main__":
    main()
