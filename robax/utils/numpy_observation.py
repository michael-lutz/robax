"""Multiprocessing is difficult with jax.Array, so we use numpy arrays instead."""

from typing import Any, Mapping, TypedDict

import numpy as np
import numpy.typing as npt


class NumpyObservation(TypedDict):
    """Numpy observation"""

    images: npt.NDArray[np.uint8] | None
    """Images (num_images, height, width, 3)"""
    text: npt.NDArray[np.uint8] | None
    """Text (num_text_tokens)"""
    proprio: npt.NDArray[np.float32] | None
    """Proprioceptive features (proprio_history, num_proprio_features)"""
    action: npt.NDArray[np.float32] | None
    """Action history and/or noisy actions (action_history, num_actions)"""


def numpy_observation_from_dict(
    observation_dict: Mapping[str, npt.NDArray[Any] | None],
) -> NumpyObservation:
    """Create an observation"""
    return NumpyObservation(
        images=observation_dict.get("images", None),
        text=observation_dict.get("text", None),
        proprio=observation_dict.get("proprio", None),
        action=observation_dict.get("action", None),
    )
