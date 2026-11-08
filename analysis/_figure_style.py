"""Figure conventions of the analysis figures: panel letters, land underlay, caption files."""

from pathlib import Path

import numpy as np

LAND_COLOUR = "0.88"


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


def add_land(ax, basin_id: str) -> None:
    """Draw the basin's land polygons in light grey under the map, in the model CRS."""
    import geopandas as gpd
    from _basin_inputs import basin_paths
    from src.grid import load_grid_def

    p = basin_paths(basin_id)
    land = gpd.read_file(
        p["sfincs_grid"].parent / f"{basin_id}_land_polygons.gpkg"
    ).to_crs(load_grid_def(p["sfincs_grid"])["crs"])
    land.plot(ax=ax, color=LAND_COLOUR, edgecolor="none", zorder=0)


def map_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


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
    note = "UNSTABLE SFINCS runs (river-network level swings): " + ", ".join(labels)
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
    return note


def save_figure(fig, path, **kwargs) -> None:
    """Save the figure, replacing an existing file (a viewer may hold it open for writing)."""
    Path(path).unlink(missing_ok=True)
    fig.savefig(path, **kwargs)
