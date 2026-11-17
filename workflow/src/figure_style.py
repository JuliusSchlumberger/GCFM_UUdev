"""Style of the deliverable figures: rcParams, sizes, project colours, panel letters, saving.

Figure scripts (analysis/ and the pipeline) call `apply_style()` once and set no
rcParams themselves.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

LAND_COLOUR = "0.88"

FIGURE_WIDTH_MM = 180.0
MM_PER_INCH = 25.4
SAVE_DPI = 300

# Okabe & Ito (2008), colour-blind safe; default cycle for categories.
OKABE_ITO = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

# Flood drivers, keyed by the scenario-name prefix (coast_100_03SLR -> "coast").
EVENT_STYLE = {
    "coast": dict(color=OKABE_ITO[5], linestyle="-", marker="o"),
    "river": dict(color=OKABE_ITO[1], linestyle="--", marker="s"),
    "compound": dict(color=OKABE_ITO[7], linestyle=":", marker="^"),
}
EVENT_LABELS = {
    "coast": "Coastal flood",
    "river": "River flood",
    "compound": "Compound flood",
}

# River water-surface options of rule 10 (analysis/surface_options.py), keyed by
# the option name there; black and two Okabe-Ito colours not used for flood drivers.
SURFACE_OPTION_STYLE = {
    "fathom_two_scale": dict(
        color="#000000", linestyle="-", label="FathomDEM, two-scale fit"
    ),
    "swot_two_scale": dict(
        color=OKABE_ITO[3], linestyle="-.", label="SWOT, two-scale fit"
    ),
    "swot_single": dict(color=OKABE_ITO[2], linestyle=":", label="SWOT, single fit"),
}

APPROACH_STYLE = {
    "PRE": dict(fmt="-o", color="#534eb8", label="Modelled (re-run)"),
    "POST": dict(fmt="--s", color="#38b394", label="Postprocessed (not re-run)"),
}
MEASURE_ORDER = [
    "advance",
    "protect_closed",
    "grey_protect_open",
    "NbS_protect_open",
    "accommodate",
    "retreat",
]
# (colour, marker) per measure; colours of the adaptation wheel
# (adaptation_wheel_individ_colors.py).
MEASURE_STYLE = {
    "advance": ("#E3C03D", "o"),
    "protect_closed": ("#009c82", "s"),
    "grey_protect_open": ("#87898C", "^"),
    "NbS_protect_open": ("#9dbf17", "D"),
    "accommodate": ("#0c5aa0", "v"),
    "retreat": ("#cc4c29", "P"),
}
# (marker, colour) per implementation scale; the markers keep the scales apart
# where the red/green colours do not.
SCALE_MARKER = {"04": ("s", "#d62728"), "09": ("^", "#f0a30a"), "1": ("o", "#2ca02c")}

STYLE_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    "image.cmap": "viridis",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": SAVE_DPI,
}


def apply_style() -> None:
    """Set the project's rcParams for deliverable figures."""
    mpl.rcParams.update(STYLE_RCPARAMS)


def style_context():
    """Return a context manager with the project rcParams, for code that must not change global state."""
    return mpl.rc_context(STYLE_RCPARAMS)


def figure_size(
    height_mm: float, width_mm: float = FIGURE_WIDTH_MM
) -> tuple[float, float]:
    """Return (width, height) in inches for a figure given in millimetres."""
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def draw_land(ax, land, zorder: int = 1) -> None:
    """Draw land polygons (a GeoDataFrame in the axes' CRS) in the project's land grey."""
    if not land.empty:
        land.plot(ax=ax, color=LAND_COLOUR, edgecolor="none", zorder=zorder)


def land_legend_handle(label: str = "Land"):
    from matplotlib.patches import Patch

    return Patch(facecolor=LAND_COLOUR, edgecolor="none", label=label)


def add_scale_bar(ax, fraction: float = 0.2) -> None:
    """Draw a scale bar in the lower-left corner of a map whose axes are in metres.

    The bar length is the round number (1, 2 or 5 x 10^n m) closest below
    `fraction` of the axes width.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    target_m = (x1 - x0) * fraction
    magnitude = 10 ** np.floor(np.log10(target_m))
    length_m = max(s * magnitude for s in (1, 2, 5) if s * magnitude <= target_m)
    xs = x0 + 0.04 * (x1 - x0)
    ys = y0 + 0.05 * (y1 - y0)
    ax.plot(
        [xs, xs + length_m],
        [ys, ys],
        color="black",
        linewidth=2,
        solid_capstyle="butt",
        zorder=10,
    )
    text = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(
        xs + length_m / 2,
        ys + 0.015 * (y1 - y0),
        text,
        ha="center",
        va="bottom",
        zorder=10,
    )


def panel_letters(axes, start: int = 0, xy: tuple[float, float] = (0.0, 1.01)) -> None:
    """Label each axis (a), (b), ... just outside its top-left corner, in the order given."""
    for k, ax in enumerate(np.ravel(axes)):
        ax.text(
            xy[0],
            xy[1],
            f"({chr(ord('a') + start + k)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )


def map_projection(crs):
    """Return the cartopy projection of a projected CRS (pyproj/rasterio CRS or EPSG code)."""
    import cartopy.crs as ccrs
    from pyproj import CRS

    return ccrs.epsg(CRS.from_user_input(crs).to_epsg())


def map_axes(ax) -> None:
    """Equal aspect and full frame; on a cartopy GeoAxes a light lon/lat graticule, labelled left and bottom.

    Plain axes get no ticks (their coordinates are model metres).
    """
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
    if hasattr(ax, "gridlines"):
        import cartopy.crs as ccrs
        import matplotlib.ticker as mticker

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.3,
            color="0.6",
            alpha=0.6,
        )
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": mpl.rcParams["xtick.labelsize"]}
        gl.xlocator = mticker.MaxNLocator(5)
        gl.ylocator = mticker.MaxNLocator(5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def map_figure_height_mm(
    x_span: float,
    y_span: float,
    width_mm: float = FIGURE_WIDTH_MM,
    map_fraction: float = 0.8,
) -> float:
    """Return the figure height that fits a map of the given spans at `map_fraction` of the width.

    The rest of the width is left for a colour bar; height is capped at 230 mm.
    """
    return min(width_mm * map_fraction * y_span / x_span, 230.0)


def write_caption(figure_path, text: str) -> None:
    """Write the figure's description next to it, as <figure>.txt."""
    Path(figure_path).with_suffix(".txt").write_text(
        text.strip() + "\n", encoding="utf-8"
    )


def run_stability_path(flood_map_tif) -> Path:
    """Return the stability.csv of the SFINCS run a max_flood_depth.tif comes from.

    Baseline maps (runs/<scenario>/visuals/) and post maps (derived from the
    baseline run) point at runs/<scenario>/sfincs/; pre maps at their own run.
    """
    folder = Path(flood_map_tif).parent
    if folder.name == "visuals":
        return folder.parent / "sfincs" / "stability.csv"
    if folder.parent.name == "pre":
        return folder / "sfincs" / "stability.csv"
    if folder.parent.name == "post":
        return folder.parents[2] / "sfincs" / "stability.csv"
    raise ValueError(f"not a run's max_flood_depth.tif: {flood_map_tif}")


def unstable_flood_maps(flood_maps: dict) -> list[str]:
    """Return the labels of {label: max_flood_depth.tif} whose run (or spin-up) is unstable.

    Runs without a stability.csv (made before the check) are reported on stdout.
    """
    import pandas as pd

    unstable = []
    for name, tif in flood_maps.items():
        path = run_stability_path(tif)
        if not path.exists():
            print(f"  stability unchecked (no {path})")
            continue
        row = pd.read_csv(path).iloc[0]
        if bool(row["unstable"]) or bool(row.get("spinup_unstable", False)):
            unstable.append(name)
    return unstable


def unstable_metric_rows(metrics) -> list[str]:
    """Return 'scenario method strategy' of the unstable rows of a metrics_comparison table."""
    if "unstable" not in metrics:
        print("  stability unchecked (metrics without an 'unstable' column)")
        return []
    rows = metrics[metrics["unstable"].fillna(False).astype(bool)]
    return [
        " ".join(
            str(v) for v in (r.scenario, r.method, r.strategy) if v != "no strategy"
        )
        for r in rows.itertuples()
    ]


def flag_unstable(fig, labels: list[str]) -> str | None:
    """Write a red footnote naming the unstable SFINCS runs a figure draws on; return it."""
    if not labels:
        return None
    note = "Check stability (river-network level swings): " + ", ".join(labels)
    warning_footnote(fig, note)
    return note


def warning_footnote(fig, note: str) -> None:
    """Write a red warning along the figure's bottom edge (PNG and PDF alike)."""
    # Inside the figure's bottom edge, so saves without bbox_inches="tight" keep it.
    fig.text(
        0.5,
        0.003,
        note,
        ha="center",
        va="bottom",
        color="#c0392b",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(fig, path, *, pdf: bool = True, **kwargs) -> None:
    """Save the figure as PNG (300 dpi unless `dpi` is given) and, by default, a PDF beside it.

    Existing files are deleted first (a viewer may hold them open for writing);
    writing goes through `src.plots.save_figure`, which retries brief file locks.
    The figure stays open.
    """
    import logging

    from src.plots import save_figure as save_with_retry

    # PDF font subsetting logs every glyph at INFO.
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    kwargs.setdefault("dpi", SAVE_DPI)
    paths = [Path(path)]
    if pdf and paths[0].suffix.lower() != ".pdf":
        paths.append(paths[0].with_suffix(".pdf"))
    for p in paths:
        p.unlink(missing_ok=True)
        save_with_retry(fig, p, close=False, **kwargs)
