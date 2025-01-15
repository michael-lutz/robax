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
