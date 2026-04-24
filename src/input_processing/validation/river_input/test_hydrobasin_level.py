from src.input_processing.utils.preprocess_01_ut_model_domains import create_model_domains
from src.input_processing.config.loader import config
from pathlib import Path


def compare_pfafstett_lvls():
    different_lvls = ['river_basins_lvl04', 'river_basins_lvl05', 'river_basins_lvl06', 'river_basins_lvl07']
    # Check if all relevant files exist
    for lvl in different_lvls:
        if Path(config['filepaths'][lvl]).is_file():
            continue
        else:
            create_model_domains(config['filepaths'][f'out_{lvl}'])

    # Extract river source positions
