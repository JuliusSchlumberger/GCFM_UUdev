import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    path : str
        Path to YAML config

    Returns
    -------
    dict
        Configuration dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


config = load_config("../src/input_processing/config/decisions.yaml")
print("Config file created")
print(config)
