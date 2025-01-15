"""Observation TypedDict"""

from typing import Dict, TypedDict

import jax


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
    observation_dict: Dict[str, jax.Array | None],
) -> Observation:
    """Create an observation"""
    return Observation(
        images=observation_dict.get("images", None),
        text=observation_dict.get("text", None),
        proprio=observation_dict.get("proprio", None),
        action=observation_dict.get("action", None),
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
