"""Shared logging configuration for Snakemake workflow scripts."""

from __future__ import annotations

import logging


def setup_logging(log_path: str, verbose: bool = False) -> logging.Logger:
    """Configure file logging for a Snakemake script and return the root logger.

    verbose adds timestamps and level names; otherwise only the message is written.
    """
    fmt = "%(asctime)s %(levelname)s  %(message)s" if verbose else "%(message)s"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format=fmt,
        force=True,  # an import may already have called basicConfig
    )
    return logging.getLogger()
