"""Map FLOPROS flood protection standards by country, for river and coast.

Values are design return periods (years): the median of a country's sub-national
FLOPROS units per hazard, on Natural Earth "admin_0_countries" (via cartopy,
cached after the first run). Needs only the raw FLOPROS table, no basin run.
Countries without a value are grey, not given the per-basin default of
src.protection_levels.identify_dominant_protection.

Usage:
    conda run -n hmt_sfincs_dev python tests/plot_global_protection_levels.py
"""

import logging
import sys
from pathlib import Path

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.io import load_catalogue, raw_input_path
from src.plots import plot_global_protection_map
from src.protection_levels import load_flopros_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = REPO_ROOT / "figs" / "protection_levels"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

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

log.info("Fetching Natural Earth country boundaries (50m, cached after first run)...")
countries_path = shpreader.natural_earth(
    resolution="50m", category="cultural", name="admin_0_countries"
)
countries_gdf = gpd.read_file(countries_path)
log.info(f"Loaded {len(countries_gdf)} country polygon(s)")

riverine_path = FIGS_DIR / "global_protection_levels_riverine.png"
coastal_path = FIGS_DIR / "global_protection_levels_coastal.png"
plot_global_protection_map(
    flopros_df, countries_gdf, str(riverine_path), str(coastal_path)
)
log.info(f"Plots written: {riverine_path}, {coastal_path}")
