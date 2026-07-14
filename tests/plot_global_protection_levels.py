"""
plot_global_protection_levels.py — Global choropleth maps of existing flood
protection standards (design return period, years) by country, for river and
coastal hazards separately, from the FLOPROS database.

This is standalone: unlike the per-basin get_protection_levels rule (04), it
does not depend on any basin/delta having been processed by the pipeline --
it only reads the raw FLOPROS table (data_catalogue "protection_levels_flopros")
directly, aggregates it to country level (median of a country's sub-national
FLOPROS units per hazard), and renders it against world country boundaries
(Natural Earth "admin_0_countries", fetched via cartopy -- cached locally
after the first run). Countries with no FLOPROS value for a hazard are shown
grey, not defaulted to some assumed protection level (that default/fallback
behaviour belongs to the per-basin forcing correction in
src.protection_levels.identify_dominant_protection, not to this diagnostic).

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
