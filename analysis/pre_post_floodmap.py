"""Map PRE and POST max flood depth of each strategy side by side, with agreement.

One figure per event and measure, one row per scale in SCALES: modelled (PRE)
and postprocessed (POST) depth on the same colour scale, plus a panel of where
the two agree. PRE is the reference (definitions and scores as in
pre_post_heatmap.py):

    hit                wet in PRE and in POST
    miss               wet in PRE, dry in POST  (flooding POST does not account for)
    false alarm        dry in PRE, wet in POST  (flooding POST adds that PRE lacks)
    protected in both  wet in the no-adaptation baseline, dry in PRE and in POST

Dry-in-both is only counted (and coloured) where the no-adaptation baseline
floods; land that floods in none of the three maps stays grey.

Output: <results_dir>/2433835/figures/analysis/90_pre_post_floodmap_<event>_<measure>.png
"""

import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pre_post_heatmap import (
    BASIN_ID,
    EVENT_TITLES,
    EVENTS,
    MEASURE_LABELS,
    MEASURES,
    THRESHOLD,
    basin_paths,
    contingency,
    land_mask,
    wet_map,
)

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.plots import read_raster_reprojected_for_plot  # noqa: E402

# one figure row per scale, top to bottom
SCALES = ["04", "09", "1"]
SCALE_LABELS = {
    "04": "Highly ineffective",
    "09": "Marginally ineffective",
    "1": "Full potential",
}
VMAX = 3.0  # m, top of the colour scale; deeper cells (e.g. the river channel) saturate
# map window (left, bottom, right, top) in degrees, cropped to the Ebro delta so the
# figure fits an A4 page; None = the flood maps' full extent
ZOOM = (0.57, 40.53, 0.90, 40.82)

PANELS = [("PRE", "Modelled approach"), ("POST", "Post-processed approach")]
# hit, miss, false alarm, protected in both
AGREE_COLORS = ["#5b9bd5", "#e66c15", "#e9e769", "#85c97f"]


def tif_path(runs, event, method, strategy):
    """Return the max flood depth raster of one adapted run."""
    return (
        runs / event / "adaptation" / method.lower() / strategy / "max_flood_depth.tif"
    )


def plot_event_measure(
    runs, domain_dir, event, event_title, measure, measure_label, land_cells
):
    """Write one figure: PRE, POST and their agreement for every scale of `measure`."""
    strategies = [f"{measure}_{s}" for s in SCALES]
    missing = [
        s
        for s in strategies
        for method, _ in PANELS
        if not tif_path(runs, event, method, s).exists()
    ]
    if missing:
        print(
            f"{event} {measure}: no PRE/POST map for {sorted(set(missing))} -- skipped"
        )
        return

    baseline_tif = runs / event / "visuals" / "max_flood_depth.tif"
    wet_base, _ = wet_map(baseline_tif)
    scores = {}
    for strategy in strategies:
        wet_pre, transform = wet_map(tif_path(runs, event, "PRE", strategy))
        wet_post, _ = wet_map(tif_path(runs, event, "POST", strategy))
        c = contingency(wet_pre, wet_post, land_cells & wet_base, transform)
        del wet_pre, wet_post
        scores[strategy] = c
        print(
            f"{event} {strategy}: hits {c['hits_km2']:.2f} km2, "
            f"misses {c['misses_km2']:.2f} km2, "
            f"false alarms {c['false_alarms_km2']:.2f} km2, "
            f"protected in both {c['protected_both_km2']:.2f} km2 | "
            f"CSI {c['CSI']:.2f}, agree {c['agree']:.2f}, HSS {c['HSS']:.2f}"
        )

    # maps, reprojected for plotting
    arrays = {}  # (strategy, method) -> depth array
    # strategy -> 1 hit / 2 miss / 3 false alarm / 4 protected in both, rest masked
    agreement = {}
    # baseline first: its bounds (left, bottom, right, top) are reused so all align
    base, (left, right, bottom, top) = read_raster_reprojected_for_plot(
        str(baseline_tif)
    )
    bounds = (left, bottom, right, top)
    wet_b = base > THRESHOLD
    for strategy in strategies:
        for method, _ in PANELS:
            arr, _ = read_raster_reprojected_for_plot(
                str(tif_path(runs, event, method, strategy)), dst_bounds=bounds
            )
            arrays[strategy, method] = arr

        wet_p = arrays[strategy, "PRE"] > THRESHOLD
        wet_q = arrays[strategy, "POST"] > THRESHOLD
        agree = np.zeros(wet_p.shape, dtype=np.uint8)
        agree[wet_p & wet_q] = 1
        agree[wet_p & ~wet_q] = 2
        agree[~wet_p & wet_q] = 3
        agree[~wet_p & ~wet_q & wet_b] = 4
        agreement[strategy] = np.ma.masked_equal(agree, 0)  # never wet -> transparent

    margin = max(right - left, top - bottom) * 0.03
    if ZOOM is None:
        view = (left - margin, bottom - margin, right + margin, top + margin)
    else:
        view = ZOOM
    land = gpd.read_file(domain_dir / f"{BASIN_ID}_land_polygons.gpkg", bbox=view)
    rivers = gpd.read_file(domain_dir / f"{BASIN_ID}_river_network_clean.gpkg")
    if rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(
        len(strategies),
        3,
        figsize=(7.5, 2.2 * len(strategies)),  # just under A4 text width
        sharex=True,
        sharey=True,
        facecolor="white",
        layout="constrained",
        squeeze=False,
    )
    for letter, row, strategy in zip("abcdefghij", axes, strategies):
        c = scores[strategy]
        row[0].text(
            0.03,
            0.97,
            f"({letter})",
            transform=row[0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            zorder=5,
        )
        for ax in row:
            land.plot(
                ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1
            )

        for ax, (method, _) in zip(row[:2], PANELS):
            im = ax.imshow(
                arrays[strategy, method],
                cmap="Blues",
                vmin=0,
                vmax=VMAX,
                extent=(left, right, bottom, top),
                origin="upper",
                aspect="auto",
                zorder=2,
            )

        row[2].imshow(
            agreement[strategy],
            cmap=ListedColormap(AGREE_COLORS),
            norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
            extent=(left, right, bottom, top),
            origin="upper",
            aspect="auto",
            zorder=2,
        )
        row[2].set_title(f"Agreement: {c['agree']:.0%} of cells")
        # blank title: same headroom as the agreement panel, so the rows line up
        for ax in row[:2]:
            ax.set_title(" ")
        row[0].set_ylabel("Latitude (°)")
        scale = strategy.rsplit("_", 1)[1]
        row[0].annotate(
            SCALE_LABELS.get(scale, scale),
            xy=(-0.42, 0.5),
            xycoords="axes fraction",
            fontsize=10,
            rotation=90,
            ha="center",
            va="center",
        )

    for ax, (_, title) in zip(axes[0, :2], PANELS):
        ax.set_title(title)
    # horizontal colorbar below the first two columns: an inset of the bottom-left
    # panel in its axes-fraction units, so ~2.1 wide spans both depth columns
    cax = axes[-1, 0].inset_axes([0.3, -0.5, 1.5, 0.05])
    fig.colorbar(
        im, cax=cax, orientation="horizontal", extend="max", label="Max flood depth (m)"
    )
    # the agreement legend is the same for every row: one copy, under the third column
    legend = axes[-1, 2].legend(
        handles=[
            Patch(facecolor=AGREE_COLORS[3], label="Protected in both"),
            Patch(facecolor=AGREE_COLORS[0], label="Hit: flooded in both"),
            Patch(facecolor=AGREE_COLORS[1], label="Miss: modelled only"),
            Patch(facecolor=AGREE_COLORS[2], label="False alarm: post-processed only"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        frameon=False,
        fontsize=7,
    )
    # colorbar and legend kept out of the constrained layout: reserving room for them
    # under some columns only would shift those columns out of line with the others
    below = [cax, legend]
    for artist in below:
        artist.set_in_layout(False)

    for ax in axes.flat:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
        ax.set_xlim(view[0], view[2])
        ax.set_ylim(view[1], view[3])
        ax.set_aspect("equal")
        ax.locator_params(nbins=4)
        ax.label_outer()
    for ax in axes[-1]:
        ax.set_xlabel("Longitude (°)")
    fig.suptitle(f"{event_title}: {measure_label}")

    # lay out without them, freeze that layout, then count them back in so the
    # tight crop at save keeps them
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for artist in below:
        artist.set_in_layout(True)

    out = analysis_path(BASIN_ID, __file__, f"{event}_{measure}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    """Write one PRE | POST | agreement figure per event and measure."""
    script_parser(__doc__).parse_args()

    # results_dir is a site setting; see config/site.example.yml.
    basin_root = site_path("results_dir") / BASIN_ID
    paths = basin_paths(basin_root)
    runs = paths["runs"]
    domain_dir = basin_root / "preprocessing_inputs" / "domain"

    land_cells = None  # one grid for every run
    for event, event_title in zip(EVENTS, EVENT_TITLES):
        for measure, measure_label in zip(MEASURES, MEASURE_LABELS):
            first = tif_path(runs, event, "PRE", f"{measure}_{SCALES[0]}")
            if land_cells is None and first.exists():
                land_cells = land_mask(first, paths["dep_subgrid"], paths["sea_mask"])
            if land_cells is None:
                print(f"{event} {measure}: no PRE map {first.name} -- skipped")
                continue
            plot_event_measure(
                runs, domain_dir, event, event_title, measure, measure_label, land_cells
            )


if __name__ == "__main__":
    main()
