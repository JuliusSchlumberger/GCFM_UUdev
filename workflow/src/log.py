"""Shared logging configuration for Snakemake workflow scripts."""

from __future__ import annotations

import logging


def setup_logging(log_path: str, verbose: bool = False) -> logging.Logger:
    """
    Configure file logging for a Snakemake script and return the root logger.

    Args:
        log_path:  Path to the Snakemake log file (snakemake.log[0]).
        verbose:   If True, include timestamps and level names in each line.
                   If False (default), write only the message — cleaner for
                   quick testing and inspection.

    Returns:
        Configured logging.Logger instance.

    Usage (in any script):
        from src.log import setup_logging
        log = setup_logging(snakemake.log[0])
        log.info("Step complete")
    """
    fmt = "%(asctime)s %(levelname)s  %(message)s" if verbose else "%(message)s"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format=fmt,
        force=True,  # overrides any earlier basicConfig call (e.g. from imports)
    )
    return logging.getLogger()
