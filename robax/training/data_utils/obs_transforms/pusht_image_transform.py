"""Pusht implements a dataloader for the Pusht dataset."""

from typing import Any, Dict

import jax
import numpy as np

from robax.training.data_utils.obs_transforms.base_train_obs_transform import (
    BaseTrainObsTransform,
)
from robax.utils.image_utils import interpolate_image_jax
from robax.utils.observation import Observation


class PushTImageTransform(BaseTrainObsTransform):
    """A class that contains a set of transformations to apply to the Pusht dataset."""

    @staticmethod
    def format_obs_cpu(horizon: Dict[str, Any]) -> Dict[str, Any]:
        """Create a proprioception feature from the state and environment state."""

        out = {}
        normalizer = np.array([512])[None, :]  # ranges from 0 to 512
        out["proprio"] = horizon["observation.state"] / normalizer
        out["action"] = horizon["action"] / normalizer
        out["images"] = np.array(horizon["observation.image"])
        return out

    @staticmethod
    @jax.jit
    def transform_obs_gpu(horizon: Observation) -> Observation:
        """Apply the transformation to the horizon."""
        assert horizon["images"] is not None, "Images are None"
        out = horizon.copy()  # shallow copy, doesn't modify the original horizon
        out["images"] = interpolate_image_jax(horizon["images"], (224, 224))
        # TODO: add augmentation here...
        return out
