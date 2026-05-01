"""Workflow for building a SFINCS model for a configured delta basin."""

from src.modelling_framework.utils.main_ut_01_build_model import (
    configure_model,
    create_grid,
    initialize_model,
)
from src.modelling_framework.utils.plotting import save_fig
from src.utils.config_loader import load_config

_MODEL_CONFIGS = load_config("../src/modelling_framework/config/decision_modelling.yml")
_BASE_MODEL_INPUTS = "../src/modelling_framework/config/base_model_inputs.yml"

_ROOTPATH = _MODEL_CONFIGS["file_paths"]["rootpath"]
_GRIDRES = _MODEL_CONFIGS["model_params"]["grid_resolution"]


def build_model(
    delta_basin_id: int = 4267691,
    data_libs: list[str] = [_BASE_MODEL_INPUTS],  # noqa: B006
    root_path: str = _ROOTPATH,
    grid_resolution: int = _GRIDRES,
    debug_plotting: bool = True,
) -> None:
    """Build a SFINCS model for a specific delta basin.

    Runs the full model-building pipeline: configures the data catalog,
    initialises the SFINCS model, creates the grid, and sets up the
    elevation component.

    Args:
        delta_basin_id: Numeric ID of the target delta basin as defined in
            the data catalog. Defaults to 4267691.
        data_libs: List of data catalog YAML paths to load. Defaults to
            ``[_BASE_MODEL_INPUTS]``.
        root_path: Root directory where the model files will be written.
            Defaults to the path configured in ``decision_modelling.yml``.
        grid_resolution: Grid cell size in metres. Defaults to the value
            configured in ``decision_modelling.yml``.
        debug_plotting: If True, diagnostic plots are saved at each pipeline
            step. Defaults to True.
    """
    catalog, delta, logger = configure_model(
        delta_basin_id, data_libs, root_path, debug_plotting
    )

    sf = initialize_model(data_libs, root_path)

    create_grid(sf, delta, grid_resolution, debug_plotting, logger)

    elevation_list = [
        {"elevation": "merit", "zmin": 0.001},
        {"elevation": "gebco"},
    ]
    sf.elevation.create(elevation_list=elevation_list)
    fig, ax = sf.plot_basemap(variable="dep", bmap="sat", plot_region=True)
    save_fig(fig, "03_elevation")
