"""Write a basin's adaptation line files from its baseline protection weir.

Needs the basin's rule-13 skeleton; rerun whenever the grid changes (domain
polygon, rotation, resolution): the baseline weir follows the grid's cell edges,
and levee measures replace only baseline segments within 2 m of their
`locations` lines. Not part of the Snakemake workflow: check the output, then
point config/adaptation_strategies.yml at it.

Writes into <out> (relative to GCFM_RAW_DATA_ROOT unless absolute):

    coastal_protection_weir_PC.geojson  protect_closed: coast + mouth closure(s)
    coastal_protection_weir_PO.geojson  grey_protect_open coastal_levee: coast only
    river_protection_weir_PO.geojson    grey_protect_open river_levee: river + estuary banks
    pump_river_mouth.geojson            pumps: river centreline across each closure, inland -> sea
    adaptation_lines.gpkg               everything above + the labelled baseline weir, for QGIS
    adaptation_lines.png                overview + zoom on every mouth
    build_info.json                     source grid, counts, per-mouth report

Usage:
    conda run -n hmt_sfincs_dev python tools/build_adaptation_lines.py 2433835 adaptation_ebro/auto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from hydromt_sfincs import SfincsModel  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.adaptation_lines import (  # noqa: E402
    KIND_COAST,
    KIND_ESTUARY,
    KIND_RIVER,
    estuary_and_seal,
    label_weir_kinds,
    mouth_closures,
    protected_land,
    relabel_estuary,
)
from src.landuse import SEA_CODE  # noqa: E402
from src.protection_weir import GridArrays, ocean_linked_wetland_mask  # noqa: E402
from src.river_burn import build_channel_mask_regular  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

KIND_COLOURS = {KIND_COAST: "black", KIND_ESTUARY: "0.6", KIND_RIVER: "purple"}


def _read_grid_raster(path: Path, like=None):
    with rasterio.open(path) as src:
        arr, transform, crs, nodata = src.read(1), src.transform, src.crs, src.nodata
    if like is not None:
        shape, ref = like
        if arr.shape != shape or not transform.almost_equals(ref, precision=1e-6):
            raise ValueError(
                f"{path} is not on the model grid ({arr.shape} vs {shape})"
            )
    return arr, transform, crs, nodata


def _plot(path, weir, closure, pump, estuary, seaward, transform, report, basin_id):
    mouths = [r for r in report if r.get("closed")]
    fig, axes = plt.subplots(
        1, 1 + len(mouths), figsize=(7.5 * (1 + len(mouths)), 7.5), squeeze=False
    )
    axes = axes[0]
    for kind, colour in KIND_COLOURS.items():
        part = weir[weir["kind"] == kind]
        if not part.empty:
            part.plot(ax=axes[0], color=colour, lw=1)
    for ax in axes:
        if not closure.empty:
            closure.plot(ax=ax, color="red", lw=3)
        if not pump.empty:
            pump.plot(ax=ax, color="blue", lw=2.5)
        ax.set_aspect("equal")
    axes[0].set_title(f"{basin_id}: baseline weir by kind, closure(s), pump(s)")
    h, w = seaward.shape
    for ax, entry in zip(axes[1:], mouths):
        x, y = entry["anchor"]
        half = max(1.5 * entry["crossing_m"], 1000.0)
        for r in range(h):
            for c in range(w):
                if not seaward[r, c]:
                    continue
                corners = [
                    transform * (c, r),
                    transform * (c + 1, r),
                    transform * (c + 1, r + 1),
                    transform * (c, r + 1),
                ]
                if (
                    abs(corners[0][0] - x) < half + 200
                    and abs(corners[0][1] - y) < half + 200
                ):
                    ax.add_patch(
                        MplPolygon(
                            corners,
                            fc="#fdd0a2" if estuary[r, c] else "#c6dbef",
                            ec="none",
                        )
                    )
        for kind, colour in KIND_COLOURS.items():
            part = weir[weir["kind"] == kind]
            if not part.empty:
                part.plot(ax=ax, color=colour, lw=2 if kind == KIND_COAST else 1.5)
        closure.plot(ax=ax, color="red", lw=3)
        pump.plot(ax=ax, color="blue", lw=2.5)
        ax.set_xlim(x - half, x + half)
        ax.set_ylim(y - half, y + half)
        ax.set_title(
            f"mouth of reach {entry['reach_id']}: {entry['crossing_m']:.0f} m crossing, "
            f"{entry['tilt_deg_from_perpendicular']:+.0f} deg, sealed={entry['sealed']}"
        )
    axes[-1].legend(
        handles=[
            Line2D([], [], color="black", lw=2, label="coast (raised: PC + PO)"),
            Line2D(
                [],
                [],
                color="0.6",
                lw=1.5,
                label="estuary bank (PC: baseline, PO: river_levee)",
            ),
            Line2D([], [], color="purple", lw=1.5, label="river bank (river_levee)"),
            Line2D([], [], color="red", lw=3, label="mouth closure (PC)"),
            Line2D([], [], color="blue", lw=2.5, label="pump line (intake -> outlet)"),
            Line2D([], [], color="#fdd0a2", lw=8, label="sea cells cut off = estuary"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Build and write the adaptation lines for one basin."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("basin_id")
    parser.add_argument(
        "out", help="output folder, relative to GCFM_RAW_DATA_ROOT unless absolute"
    )
    parser.add_argument(
        "--basin-dir", help="basin results folder, if not GCFM_RESULTS_DIR/<basin_id>"
    )
    args = parser.parse_args()
    bid = args.basin_id

    results = (
        Path(args.basin_dir)
        if args.basin_dir
        else Path(os.environ["GCFM_RESULTS_DIR"]) / bid
    )
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = results / "preprocessing_inputs" / "domain"
    skeleton = results / "sfincs_skeleton"
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yml").read_text())
    rdm = config["river_processing"]["river_depth_modelling"]

    landuse, transform, crs, _ = _read_grid_raster(
        domain / f"{bid}_landuse_on_grid.tif"
    )
    mask, *_ = _read_grid_raster(
        skeleton / "gis" / "mask.tif", like=(landuse.shape, transform)
    )
    valid = mask > 0
    # The weir as the model holds it (and as levee measures merge against):
    # hydromt_sfincs drops rule-10 segments outside the model region.
    sf = SfincsModel(root=str(skeleton), mode="r")
    sf.weirs.read()
    weir = gpd.GeoDataFrame(
        geometry=[
            LineString([c[:2] for c in g.coords]) for g in sf.weirs.data.geometry
        ],
        crs=sf.crs,
    ).to_crs(crs)
    rivers = gpd.read_file(domain / f"{bid}_river_network_clean.gpkg").to_crs(crs)
    grid_def = json.loads((domain / f"{bid}_sfincs_grid.json").read_text())

    seaward = landuse == SEA_CODE
    if rdm.get("unprotected_ocean_wetlands", False):
        grid = GridArrays(
            crs,
            np.zeros(landuse.shape, np.float32),
            valid,
            float(np.hypot(transform.a, transform.d)),
            transform,
        )
        seaward = seaward | ocean_linked_wetland_mask(landuse, seaward, grid)
    channel = build_channel_mask_regular(rivers, "width", landuse.shape, transform)

    kinds, kind_counts = label_weir_kinds(
        weir, seaward, channel, valid, transform, int(rdm["min_component_cells"])
    )
    log.info(f"Baseline weir labels: {kind_counts}")
    land = protected_land(weir, seaward, valid, transform)
    closures, pumps, report = mouth_closures(rivers, land, transform)
    estuary = estuary_and_seal(weir, closures, report, seaward, valid, transform)
    kinds = relabel_estuary(weir, kinds, estuary, transform)
    weir = weir.assign(kind=kinds)
    log.info(
        f"After closure: {pd.Series(kinds).value_counts().to_dict()}, "
        f"{int(estuary.sum())} sea cell(s) cut off as estuary"
    )

    closure = (
        gpd.GeoDataFrame(
            [
                {
                    "reach_id": c["reach_id"],
                    "kind": "closure",
                    "geometry": c["geometry"],
                }
                for c in closures
            ],
            geometry="geometry",
            crs=crs,
        )
        if closures
        else gpd.GeoDataFrame({"reach_id": [], "kind": []}, geometry=[], crs=crs)
    )
    pump = (
        gpd.GeoDataFrame(pumps, geometry="geometry", crs=crs)
        if pumps
        else gpd.GeoDataFrame({"reach_id": []}, geometry=[], crs=crs)
    )

    lines = weir[["kind", "geometry"]]
    outputs = {
        "coastal_protection_weir_PC.geojson": pd.concat(
            [lines[lines["kind"] == KIND_COAST], closure[["kind", "geometry"]]],
            ignore_index=True,
        ),
        "coastal_protection_weir_PO.geojson": lines[lines["kind"] == KIND_COAST],
        "river_protection_weir_PO.geojson": lines[
            lines["kind"].isin([KIND_RIVER, KIND_ESTUARY])
        ],
        "pump_river_mouth.geojson": pump,
    }
    for name, gdf in outputs.items():
        gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs).to_file(
            out_dir / name, driver="GeoJSON"
        )
        log.info(f"Written: {out_dir / name} ({len(gdf)} feature(s))")
    gpkg = out_dir / "adaptation_lines.gpkg"
    weir.to_file(gpkg, layer="baseline_weir_labelled", driver="GPKG")
    if not closure.empty:
        closure.to_file(gpkg, layer="closure", driver="GPKG")
    if not pump.empty:
        pump.to_file(gpkg, layer="pump", driver="GPKG")
    _plot(
        out_dir / "adaptation_lines.png",
        weir,
        closure,
        pump,
        estuary,
        seaward,
        transform,
        report,
        bid,
    )

    unsealed = [
        r["reach_id"] for r in report if r.get("closed") and not r.get("sealed")
    ]
    for r in report:
        r.pop("river_check_point", None)
    info = {
        "basin_id": bid,
        "built": datetime.now().isoformat(timespec="seconds"),
        "source_weir": str(skeleton / "sfincs.weir"),
        "grid": {
            k: grid_def[k]
            for k in ("x0", "y0", "dx", "dy", "mmax", "nmax", "rotation")
            if k in grid_def
        },
        "baseline_weir_kinds": pd.Series(kinds).value_counts().to_dict(),
        "facing_small_island": kind_counts["facing_small_island"],
        "unresolved_as_river": kind_counts["unresolved_as_river"],
        "estuary_cells": int(estuary.sum()),
        "mouths": report,
    }
    (out_dir / "build_info.json").write_text(json.dumps(info, indent=2))
    log.info(
        f"Written: {out_dir / 'adaptation_lines.png'}, {out_dir / 'build_info.json'}"
    )
    if unsealed:
        log.error(
            f"NOT SEALED: outlet reach(es) {unsealed} -- check the plot before using the PC file"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
