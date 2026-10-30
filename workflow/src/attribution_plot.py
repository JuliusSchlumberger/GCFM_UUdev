"""Classify a flood map by dominant source (river, coastal, compound, permanent water).

Uses existing max_flood_depth.tif rasters of the river and coastal counterpart
scenarios (00_common.smk attribution_counterparts: one driver at the target
scenario's own level, the other at mean conditions, same slr_m) and the spin-up (mean river discharge, calm sea). Called by rule attribution_mask
(17a_attribution_mask.py).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import rasterio
import rioxarray as rxr
from hydromt_sfincs import SfincsModel
from matplotlib.colors import BoundaryNorm, ListedColormap
from src.plots import add_land_background_to_geoaxes, save_figure


def generate_attribution_maps(
    base_root: Path,
    attribution_runs: list[tuple[str, Path, Path, Path, Path, Path]],
    land_mask_path: str | Path,
    threshold: float = 0.05,
    colors_cat: list[str] = ["#d1c740", "#2168c5", "#b063c0", "#7a9cb1"],
    labels_cat: list[str] = [
        "River",
        "Coastal",
        "Compound",
        "Permanent water",
    ],
    data_libs: list[str] | None = None,
) -> None:
    """Write attribution_mask.tif and its map for each (label, river, coastal, spinup, out_dir, png) run.

    Classes: 1 river, 2 coastal, 3 compound, 4 permanent water. Cells wet in
    the spin-up (e.g. the perennial channel, which every event run restarts
    from) are class 4 and override the others. threshold is the depth (m)
    above which a cell counts as flooded in each input raster. base_root is
    opened read-only for the basemap only; land_mask_path
    ({basin}_land_mask_on_grid.gpkg) is the background, no web tiles. Runs with
    a missing input TIF are skipped.
    """
    cmap_attr = ListedColormap(colors_cat)
    norm_attr = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_attr.N)

    mod_ref = SfincsModel(root=str(base_root), data_libs=data_libs, mode="r")

    for (
        label,
        river_tif,
        coastal_tif,
        spinup_tif,
        out_folder,
        out_png,
    ) in attribution_runs:
        river_tif, coastal_tif, spinup_tif, out_folder, out_png = (
            Path(river_tif),
            Path(coastal_tif),
            Path(spinup_tif),
            Path(out_folder),
            Path(out_png),
        )
        if (
            not river_tif.exists()
            or not coastal_tif.exists()
            or not spinup_tif.exists()
        ):
            print(f"  Skipping '{label}': TIF(s) not found.")
            continue

        print(f"\n--- Attribution: {label} ---")

        with rasterio.open(river_tif) as src:
            river = src.read(1)
            profile = src.profile.copy()
        with rasterio.open(coastal_tif) as src:
            coastal = src.read(1)
        with rasterio.open(spinup_tif) as src:
            spinup = src.read(1)

        r_flood = river > threshold
        c_flood = coastal > threshold
        s_flood = spinup > threshold

        attr_mask = np.zeros(river.shape, dtype=np.uint8)
        attr_mask[r_flood & ~c_flood] = 1
        attr_mask[~r_flood & c_flood] = 2
        attr_mask[r_flood & c_flood] = 3
        attr_mask[s_flood] = 4

        out_folder.mkdir(parents=True, exist_ok=True)

        out_tif = out_folder / "attribution_mask.tif"
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=None)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(attr_mask, 1)
        print(f"  Saved TIF: {out_tif}")

        da_attr = rxr.open_rasterio(out_tif).squeeze(drop=True).astype(float)
        # Classes 0 and 4 are transparent on the map; the TIF keeps class 4.
        da_attr = da_attr.where((da_attr > 0) & (da_attr != 4))
        da_attr.name = "attribution"

        fig, ax = mod_ref.plot_basemap(
            figsize=(10, 8),
            variable=da_attr,
            plot_bounds=False,
            plot_geoms=False,
            cmap=cmap_attr,
            norm=norm_attr,
            cbar_kwargs={"shrink": 0},
        )
        add_land_background_to_geoaxes(ax, str(land_mask_path), mod_ref.crs)
        for _ax in fig.axes:
            if _ax is not ax:
                _ax.remove()
        # Classes 1-3 (river/coastal/compound) always appear in the legend
        # regardless of whether this basin x scenario actually has any
        # pixels in them; class 4 (spin-up/permanent water) is masked out of
        # the map and gets no legend entry, since it isn't a real attribution
        # outcome.
        ax.legend(
            handles=[
                mpatches.Patch(color=colors, label=labels)
                for colors, labels in zip(colors_cat[:3], labels_cat[:3])
            ],
            loc="lower right",
            framealpha=0.9,
            fontsize=10,
        )
        ax.set_title(f"Flood source attribution – {label}")
        save_figure(fig, out_png, dpi=150, bbox_inches="tight")
        print(f"  Saved PNG: {out_png}")
