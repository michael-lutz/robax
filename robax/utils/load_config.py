"""Load config from yaml file."""

import yaml  # type: ignore

from robax.config.base_training_config import Config


def load_config(path: str) -> Config:
    """Load config from yaml file.

    Args:
        path: Path to the yaml file.

    Returns:
        Config: The config.
    """
    with open(path, "r") as file:
        config: Config = yaml.safe_load(file)

    return config
