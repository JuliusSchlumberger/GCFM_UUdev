"""Logging and plotting initialisation utilities for the modelling framework."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from hydromt._utils import log


def initialize_logger(delta_basin_id: int, root_folder: str) -> logging.Logger:
    """Initialise the HydroMT logger with console and file handlers.

    Configures the HydroMT logger at INFO level, disables interactive
    matplotlib plotting, and attaches a file handler writing to
    ``hydromt_sfincs.log`` inside *root_folder*.

    Args:
        delta_basin_id: Numeric basin ID logged in the opening message,
            used to identify which delta is being processed.
        root_folder: Root directory for the model run. The log file is
            written to ``<root_folder>/hydromt_sfincs.log``.

    Returns:
        Configured :class:`logging.Logger` instance for the ``"hydromt"``
        logger.
    """
    log.initialize_logging()
    log.set_log_level(
        log_level=20
    )  # NOTSET=0-9, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50

    plt.ioff()  # Turn off interactive plotting

    log_file = Path(root_folder) / "hydromt_sfincs.log"
    log._add_filehandler(log_file)
    logger = logging.getLogger("hydromt")

    logger.info("--- Starting model setup for Basin ID: %s ---", delta_basin_id)

    return logger
