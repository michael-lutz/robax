"""Pusht implements a dataloader for the Pusht dataset."""

from typing import Any, Dict

import numpy as np

from robax.training.data_utils.obs_transforms.base_train_obs_transform import (
    BaseTrainObsTransform,
)


class PushTKeypointsTransform(BaseTrainObsTransform):
    """A class that contains a set of transformations to apply to the Pusht dataset."""

    @staticmethod
    def __call__(horizon: Dict[str, Any]) -> Dict[str, Any]:
        """Create a proprioception feature from the state and environment state."""

        out = {}
        normalizer = np.array([512])[None, :]  # ranges from 0 to 512
        out["proprio"] = (
            np.concatenate(
                [horizon["observation.state"], horizon["observation.environment_state"]], axis=-1
            )
            / normalizer
        )
        out["action"] = horizon["action"] / normalizer
        return out
