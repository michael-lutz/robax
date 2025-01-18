"""Observation TypedDict"""

from typing import Mapping, TypedDict

import jax

from robax.utils.numpy_observation import NumpyObservation


class Observation(TypedDict):
    """Observation TypedDict"""

    images: jax.Array | None
    """Images (num_images, height, width, 3)"""
    text: jax.Array | None
    """Text (num_text_tokens)"""
    proprio: jax.Array | None
    """Proprioceptive features (proprio_history, num_proprio_features)"""
    action: jax.Array | None
    """Action history and/or noisy actions (action_history, num_actions)"""


def observation_from_dict(
    observation_dict: Mapping[str, jax.Array | None],
) -> Observation:
    """Create an observation"""
    return Observation(
        images=observation_dict.get("images", None),
        text=observation_dict.get("text", None),
        proprio=observation_dict.get("proprio", None),
        action=observation_dict.get("action", None),
    )


def observation_from_numpy_observation(numpy_observation: NumpyObservation) -> Observation:
    """Convert a numpy observation to a jax observation"""
    return Observation(
        images=(
            jax.numpy.array(numpy_observation.get("images"))
            if numpy_observation.get("images") is not None
            else None
        ),
        text=(
            jax.numpy.array(numpy_observation.get("text"))
            if numpy_observation.get("text") is not None
            else None
        ),
        proprio=(
            jax.numpy.array(numpy_observation.get("proprio"))
            if numpy_observation.get("proprio") is not None
            else None
        ),
        action=(
            jax.numpy.array(numpy_observation.get("action"))
            if numpy_observation.get("action") is not None
            else None
        ),
    )


def get_batch_size(observation: Observation) -> int:
    """Get the batch size from an observation"""
    non_none_elements = [v.shape[0] for v in observation.values() if v is not None]  # type: ignore
    if not non_none_elements:
        raise ValueError("All elements in the TypedDict are None.")

    reference_batch_dim: int = non_none_elements[0]

    for element in non_none_elements:
        if element != reference_batch_dim:
            raise ValueError("Batch dimensions do not match.")

    return reference_batch_dim
