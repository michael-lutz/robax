"""Base training config.

In my opinion, YAML configs are a pain to work with because it's unclear what the types are,
implicit inheritance, and difficulty of debugging.

This is a simple config class that allows for explicit inheritance and type checking.
"""

from typing import Callable, Dict, List

import attrs


@attrs.define
class BaseDataConfig:
    """Stores configuration for the dataset and dataloader initialization."""

    # Relevant to the dataset
    repo_id: str = attrs.field()
    """Hugging Face repo ID."""
    image_transforms: Callable = attrs.field(default=None)
    """Image transforms to apply to the dataset."""
    delta_timesteps: Dict[str, List[float]] = attrs.field(default=None)
    """Delta timesteps to apply to the dataset."""

    # Relevant to the dataloader
    num_workers: int = attrs.field(default=4)
    """Number of workers to use for loading the dataset."""
    batch_size: int = attrs.field(default=32)
    """Batch size to use for loading the dataset."""


@attrs.define
class BaseModelConfig:
    """Stores configuration for model hyperparameters. Highly bespoke."""

    model_name: str = attrs.field()
    """Model name to be used in the training loop."""


@attrs.define
class BaseTrainingConfig:
    """Stores configuration for the training loop."""

    data: BaseDataConfig = attrs.field()
    """Data config."""
