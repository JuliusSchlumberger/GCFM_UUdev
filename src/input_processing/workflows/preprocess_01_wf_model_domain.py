# from src.input_processing.utils.defining_model_domain import create_model_domains
from src.input_processing.utils.preprocess_01_ut_model_domains import (
    create_model_domains,
)
from src.input_processing.config.loader import config


def create_domains_chosen_level(
    used_delta_polygons: str = config["filepaths"]["hand_picked_deltas"],
    outpath_domains: str = config["filepaths"]["new_domains"],
    outpath_mismatched: str = config["filepaths"]["mismatched_polygons"],
    outpath_subset: str = config["filepaths"]["delta_polygons_used"],
    pfaf_path: str = config["filepaths"]["river_basins_applied"],
):
    create_model_domains(
        used_delta_polygons=used_delta_polygons,
        outpath_domains=outpath_domains,
        outpath_mismatched=outpath_mismatched,
        outpath_subset=outpath_subset,
        pfaf_path=pfaf_path,
    )
