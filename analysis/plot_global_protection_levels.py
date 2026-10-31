"""Map FLOPROS flood protection standards by country, for river and coast.

Values are design return periods (years): the median of a country's sub-national
FLOPROS units per hazard, on Natural Earth "admin_0_countries" (via cartopy,
cached after the first run). Needs only the raw FLOPROS table, no basin run.
Countries without a value are grey, not given the per-basin default of
src.protection_levels.identify_dominant_protection. Writes
plot_global_protection_levels_riverine.png and _coastal.png to
<results_dir>/figures/analysis/.

Usage:
    conda run -n <env> python analysis/plot_global_protection_levels.py
"""

import logging
import sys

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from _common import REPO_ROOT, analysis_path, script_parser  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.io import load_catalogue, raw_input_path
from src.plots import _save
from src.protection_levels import load_flopros_table

log = logging.getLogger(__name__)


def plot_global_protection_map(
    flopros_df,
    countries_gdf: gpd.GeoDataFrame,
    riverine_output_path: str,
    coastal_output_path: str,
    iso_column: str = "ISO_A3_EH",
    agg: str = "median",
) -> None:
    """Plot world maps of country-level FLOPROS design protection RP (years).

    One riverine and one coastal figure on a shared log scale. Each country is
    the agg (default median) over its geounits with a value; countries with
    none stay NaN (grey), with no fallback. flopros_df needs "ISO", "Riverine",
    "Coastal". iso_column defaults to Natural Earth "ISO_A3_EH" because
    "ISO_A3" is -99 for some countries (e.g. France, Norway).
    """
    country_agg = flopros_df.groupby("ISO")[["Riverine", "Coastal"]].agg(agg)
    merged = countries_gdf.merge(
        country_agg, how="left", left_on=iso_column, right_index=True
    )

    finite_vals = merged[["Riverine", "Coastal"]].to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    max_rp_yr = max(100.0, float(finite_vals.max())) if finite_vals.size else 100.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max_rp_yr)
    cmap = plt.get_cmap("YlOrRd").copy()

    # geopandas locks equal aspect, and bbox_inches="tight" cannot crop the
    # blank axes margin, so figsize follows the 360x145 deg map aspect.
    map_aspect = (85 - -60) / (180 - -180)  # lat span / lon span
    fig_w_in = 14.0
    fig_h_in = fig_w_in * map_aspect + 1.3  # + title/colorbar headroom

    output_paths = {"Riverine": riverine_output_path, "Coastal": coastal_output_path}
    for hazard, output_path in output_paths.items():
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), constrained_layout=True)
        merged.plot(
            column=hazard,
            ax=ax,
            cmap=cmap,
            norm=norm,
            missing_kwds={"color": "#d0d0d0", "label": "No FLOPROS data"},
            edgecolor="#888888",
            linewidth=0.2,
        )
        n_with_data = int(merged[hazard].notna().sum())
        ax.set_title(
            f"{hazard} flood protection standard\n"
            f"{n_with_data}/{len(merged)} countries with FLOPROS data ({agg} of sub-national units)"
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.5
        )
        cbar.set_label("Design protection return period (years)")
        _save(fig, output_path)


def main() -> None:
    """Run the FLOPROS map rendering for river and coast."""
    script_parser(__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(REPO_ROOT / "config" / "config.yml") as fh:
        config = yaml.safe_load(fh)

    catalogue = load_catalogue(config["data_catalogue"])
    flopros_path = raw_input_path(catalogue, "protection_levels_flopros")

    log.info(f"Loading FLOPROS table: {flopros_path}")
    flopros_df = load_flopros_table(flopros_path)
    log.info(
        f"Loaded {len(flopros_df)} geounit(s), "
        f"{flopros_df['ISO'].nunique()} distinct ISO3 code(s), "
        f"Riverine: {flopros_df['Riverine'].notna().sum()} populated, "
        f"Coastal: {flopros_df['Coastal'].notna().sum()} populated"
    )

    log.info(
        "Fetching Natural Earth country boundaries (50m, cached after first run)..."
    )
    countries_path = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    countries_gdf = gpd.read_file(countries_path)
    log.info(f"Loaded {len(countries_gdf)} country polygon(s)")

    riverine_path = analysis_path(None, __file__, "riverine.png")
    coastal_path = analysis_path(None, __file__, "coastal.png")
    plot_global_protection_map(
        flopros_df, countries_gdf, str(riverine_path), str(coastal_path)
    )
    log.info(f"Plots written: {riverine_path}, {coastal_path}")


if __name__ == "__main__":
    main()
