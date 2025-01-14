"""Naive wrapper for the Lerobot dataset to make it compatible with JAX."""

from typing import Dict

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


class JaxLeRobotDataset(LeRobotDataset):
    def __getitem__(self, index: int) -> Dict[str, jax.Array]:
        data = super().__getitem__(index)
        img = jnp.array(data["observation.image"]).transpose(0, 2, 3, 1)
        proprio = jnp.array(data["observation.state"])
        action = jnp.array(data["action"])
        action = action - jnp.array([227.166, 294.710])
        action = action / jnp.array([100.304, 94.476])
        return {"image": img, "text": jnp.array([]), "proprio": proprio, "action": action}
