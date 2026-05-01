"""Shared logging configuration for all pipeline modules."""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_fname: str,
    level: int = logging.INFO,
    log_dir: Path | None = Path("logs"),
) -> logging.Logger:
    """Configure and return the module logger.

    Writes to stdout and, optionally, to a file in *log_dir*. If *log_dir*
    is None, no file is written.

    Args:
        log_fname: Logger name and stem of the log file (e.g.
            ``"roughness_from_landcover"`` → ``logs/roughness_from_landcover.log``).
        level: Logging level applied to both handlers. Defaults to
            ``logging.INFO``.
        log_dir: Directory for the log file. Created if it does not exist.
            Pass None to suppress file logging. Defaults to ``Path("logs")``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(log_fname)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{log_fname}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(level)
    return logger
