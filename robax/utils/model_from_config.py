"""Model instantiation based on config"""

import flax.linen as nn

from robax.config.base_training_config import ModelConfig


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
