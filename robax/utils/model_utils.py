"""Model instantiation based on config"""

import os
import pickle
from typing import Any, Dict

import flax.linen as nn
import yaml

from robax.config.base_training_config import Config, ModelConfig


def get_model(config: ModelConfig) -> nn.Module:
    """Barebones config-based model instantiation

    Args:
        config: The model config.

    Returns:
        The model.
    """

    if config["name"] == "pi_zero":
        from robax.model.policy.pi_zero import PiZero

        return PiZero(**config["args"])
    else:
        raise ValueError("Unknown model name in config")


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


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint from pkl file.

    Args:
        path: Path to the pkl file.

    Returns:
        Dict[str, Any]: The checkpoint.
    """
    with open(path, "rb") as file:
        checkpoint: Dict[str, Any] = pickle.load(file)

    return checkpoint


def save_checkpoint(params: Dict[str, Any], checkpoint_dir: str, epoch: int, i: int) -> None:
    """Save the model parameters to a checkpoint file.

    Args:
        params: The model parameters to save.
        checkpoint_dir: The directory where checkpoints will be saved.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{i}.pkl")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
    with open(checkpoint_path, "wb") as f:
        pickle.dump(params, f)
