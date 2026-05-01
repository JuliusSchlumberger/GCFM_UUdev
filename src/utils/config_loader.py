"""Configuration loader for the input processing pipeline.

Loads the project YAML configuration file once at import time and exposes it
as the module-level ``config`` dictionary. All other modules import ``config``
from here rather than reading the file themselves.

Example:
    >>> config = load_config("../src/input_processing/config/decisions.yaml")
    >>> print(config["CRS"]["standard"])
    4326
"""

from __future__ import annotations

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "decisions.yaml"


def load_config(path: Path | str) -> dict:
    """Load a YAML configuration file and return its contents as a dictionary.

    Args:
        path: Path to the YAML config file. Accepts either a
            :class:`~pathlib.Path` object or a plain string. Relative paths
            are resolved from the current working directory.

    Returns:
        Dictionary containing the parsed YAML configuration. The structure
        mirrors the YAML file hierarchy directly.

    Raises:
        FileNotFoundError: If no file exists at *path*.
        yaml.YAMLError: If the file exists but contains invalid YAML.

    Example:
        >>> cfg = load_config("src/input_processing/config/decisions.yaml")
        >>> print(cfg["CRS"]["standard"])
        4326
    """
    config_path: Path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# config: dict = load_config(_CONFIG_PATH)  # this does not work now that config_loader is moved to separate directory TODO: fix reference for all input_processing!
