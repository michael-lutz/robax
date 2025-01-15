"""Observation TypedDict"""

from typing import TypedDict

import jax


class Observation(TypedDict):
    """Observation TypedDict"""

    images: jax.Array
    """Images (num_images, height, width, 3)"""
    text: jax.Array
    """Text (num_text_tokens)"""
    proprio: jax.Array
    """Proprioceptive features (proprio_history, num_proprio_features)"""
    action: jax.Array
    """Action history and/or noisy actions (action_history, num_actions)"""
