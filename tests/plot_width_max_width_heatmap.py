"""
plot_width_max_width_heatmap.py — For every basin in the delta_polygons
dataset (the same source workflow/rules/00_common.smk's list_basins() uses
to build the Snakemake pipeline's BASINS list), load its cleaned river
network (river_network_clean.gpkg, rule clean_river_network) and plot a 2D
density heatmap of 'width' vs 'max_width' across all its reaches.

Diagnostic for spotting anomalous max_width values (see
src.river_network.clip_anomalous_max_width) -- reaches far above the y=x
diagonal, especially beyond the river_processing.cleaning.max_width_to_width_ratio
cap line, are the SWORD max_width data-quality issue that motivated that fix.
Both axes are log-scaled since width/max_width span orders of magnitude.

Basins without a results/{basin_id}/inputs/domain/river_network_clean.gpkg
yet (not processed through the pipeline) are skipped with a warning.

Usage:
    conda run -n hmt_sfincs_dev python tests/plot_width_max_width_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.io import catalogue_entry, load_catalogue, raw_input_path, read_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = REPO_ROOT / "figs" / "width_max_width_heatmap"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
MAX_RATIO = config["river_processing"]["cleaning"]["max_width_to_width_ratio"]

N_BINS = 40


def list_basins() -> list[int]:
    """Discover available basin ids the same way workflow/rules/00_common.smk does."""
    catalogue = load_catalogue(REPO_ROOT / config["data_catalogue"])
    dataset = catalogue_entry(catalogue, "delta_polygons")
    attribute = dataset["attributes"][0]["name"]
    deltas = read_geometry(raw_input_path(catalogue, "delta_polygons"))
    return sorted(deltas[attribute].astype(int).to_list())


def plot_basin_heatmap(
    basin_id: int, width: np.ndarray, max_width: np.ndarray, output_path: Path
) -> None:
    log_w = np.log10(width)
    log_mw = np.log10(max_width)
    lo = min(log_w.min(), log_mw.min()) - 0.1
    hi = max(log_w.max(), log_mw.max()) + 0.1
    bins = np.linspace(lo, hi, N_BINS + 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(log_w, log_mw, bins=bins, cmap="viridis", norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="reach count")

    diag = np.array([lo, hi])
    ax.plot(
        diag,
        diag,
        color="white",
        linewidth=1.0,
        linestyle="--",
        label="max_width = width",
    )
    ax.plot(
        diag,
        diag + np.log10(MAX_RATIO),
        color="red",
        linewidth=1.0,
        linestyle="--",
        label=f"max_width = {MAX_RATIO:.0f}x width",
    )

    ticks = list(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10**t:g}" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{10**t:g}" for t in ticks])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("width (m)")
    ax.set_ylabel("max_width (m)")
    ax.set_title(f"Basin {basin_id}: width vs max_width ({width.size} reaches)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    basin_ids = list_basins()
    print(f"{len(basin_ids)} basin(s) in delta_polygons")

    n_plotted = 0
    n_skipped = 0
    for basin_id in basin_ids:
        river_path = (
            RESULTS_DIR
            / str(basin_id)
            / "inputs"
            / "domain"
            / f"{basin_id}_river_network_clean.gpkg"
        )
        if not river_path.exists():
            print(f"basin {basin_id}: no river_network_clean.gpkg yet, skipping")
            n_skipped += 1
            continue

        rivers = gpd.read_file(river_path)
        valid = (
            rivers["width"].notna()
            & rivers["max_width"].notna()
            & (rivers["width"] > 0)
            & (rivers["max_width"] > 0)
        )
        width = rivers.loc[valid, "width"].to_numpy(dtype=float)
        max_width = rivers.loc[valid, "max_width"].to_numpy(dtype=float)
        n_dropped = int((~valid).sum())
        if width.size == 0:
            print(f"basin {basin_id}: no valid width/max_width pairs, skipping")
            n_skipped += 1
            continue

        out_path = FIGS_DIR / f"{basin_id}_width_max_width_heatmap.png"
        plot_basin_heatmap(basin_id, width, max_width, out_path)
        print(
            f"basin {basin_id}: {width.size} reach(es) plotted "
            f"({n_dropped} dropped: missing/non-positive width or max_width) -> {out_path}"
        )
        n_plotted += 1

    print(f"\nDone: {n_plotted} basin(s) plotted, {n_skipped} skipped")


if __name__ == "__main__":
    main()
