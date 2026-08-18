"""
attribution_plot.py -- Classify a basin x scenario's flood map by dominant
source (river / coastal / compound / spin-up "permanent water"), from
three already-computed max_flood_depth.tif rasters (no new SFINCS runs):
- a river-only counterpart scenario's own flood map
- a coastal-only counterpart scenario's own flood map
- the basin-level spin-up's own flood depth (RP=1, both drivers -- the
  permanent-water class, overrides the other three; catches cells like the
  perennial river channel that stay wet from the SAME spin-up restart every
  scenario's own event run picks up from, regardless of that scenario's own
  driver)

Counterpart scenarios are resolved by 00_common.smk's attribution_counterparts
(RP-matched against the target scenario's own river_rp/surge_rp) -- this
module itself is result-path-agnostic, driven entirely by the paths it's
given. Called once per basin x scenario by 18c_attribution_mask.py (rule
attribution_mask, 18c_attribution_mask.smk).
"""

from pathlib import Path

import numpy as np
import rasterio
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from hydromt_sfincs import SfincsModel


def generate_attribution_maps(
    base_root: Path,
    attribution_runs: list[tuple[str, Path, Path, Path, Path]],
    threshold: float = 0.05,
    colors_cat: list[str] = ["#d1c740", "#3277d3", "#b063c0", "#d8d4d4"],
    labels_cat: list[str] = [
        "River",
        "Coastal",
        "Compound",
        "Spinup (permanent water)",
    ],
    data_libs: list[str] | None = None,
    zoomlevel: int = 11,
) -> None:
    """
    Args:
        base_root        : basin skeleton root, opened read-only purely for
                            plot_basemap's own model geometry/basemap -- no
                            data_catalog lookups happen here.
        attribution_runs : list of (label, river_tif, coastal_tif, spinup_tif,
                            out_folder) tuples -- one per basin x scenario being
                            classified. spinup_tif is the basin-level spin-up's
                            own flood depth (RP=1, both drivers) -- cells
                            already wet there (e.g. the perennial river
                            channel, kept wet by the SAME spin-up restart every
                            scenario's own event run picks up from) are
                            permanent water, not event-driven flooding, and
                            override any river/coastal/compound classification.
        threshold         : depth [m] above which a cell counts as flooded
                            in each of the three input rasters.
    """
    cmap_attr = ListedColormap(colors_cat)
    norm_attr = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_attr.N)

    mod_ref = SfincsModel(root=str(base_root), data_libs=data_libs, mode="r")

    for label, river_tif, coastal_tif, spinup_tif, out_folder in attribution_runs:
        river_tif, coastal_tif, spinup_tif, out_folder = (
            Path(river_tif),
            Path(coastal_tif),
            Path(spinup_tif),
            Path(out_folder),
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
        attr_mask[r_flood & ~c_flood] = 1  # river-dominated
        attr_mask[~r_flood & c_flood] = 2  # coastal-dominated
        attr_mask[r_flood & c_flood] = 3  # compound
        attr_mask[s_flood] = 4  # spin-up/permanent water -- overrides the above

        out_folder.mkdir(parents=True, exist_ok=True)

        # Save GeoTIFF
        out_tif = out_folder / "attribution_mask.tif"
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=None)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(attr_mask, 1)
        print(f"  Saved TIF: {out_tif}")

        # Load back as georeferenced DataArray
        da_attr = rxr.open_rasterio(out_tif).squeeze(drop=True).astype(float)
        da_attr = da_attr.where(da_attr > 0)  # class 0 → NaN (transparent)
        da_attr.name = "attribution"

        fig, ax = mod_ref.plot_basemap(
            figsize=(10, 8),
            variable=da_attr,
            plot_bounds=False,
            plot_geoms=False,
            bmap="sat",
            zoomlevel=zoomlevel,
            cmap=cmap_attr,
            norm=norm_attr,
            cbar_kwargs={"shrink": 0},
        )
        for _ax in fig.axes:
            if _ax is not ax:
                _ax.remove()
        ax.legend(
            handles=[
                mpatches.Patch(color=colors, label=labels)
                for colors, labels in zip(colors_cat, labels_cat)
            ],
            loc="lower right",
            framealpha=0.9,
            fontsize=10,
        )
        ax.set_title(f"Flood source attribution – {label}")
        out_png = out_folder / "attribution_mask.png"
        fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved PNG: {out_png}")
