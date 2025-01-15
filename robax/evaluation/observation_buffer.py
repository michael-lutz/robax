"""Basic FIFO buffer for storing observations during evaluation."""

from collections import deque
from typing import Deque, Dict

import jax.numpy as jnp


class BatchedObservationBuffer:
    """Basic FIFO buffer that returns JAX array with observation history.

    TODO: Make it use delta timesteps as well...
    """

    def __init__(self, observation_sizes: Dict[str, int]):
        """Initialize the observation buffer.

        Args:
            observation_sizes: A dictionary of observation sizes.
        """
        self.buffers: Dict[str, Deque[jnp.ndarray]] = {
            key: deque(maxlen=size) for key, size in observation_sizes.items()
        }

    def add(self, observation: Dict[str, jnp.ndarray]) -> None:
        """Add an observation to the buffer. Note that every observation should have a batch dim.

        Args:
            observation: A dictionary of observations.
        """
        for key, value in observation.items():
            self.buffers[key].append(value)

    def get_observation(self) -> Dict[str, jnp.ndarray]:
        """Get the observation history as a JAX array.

        Returns:
            A dictionary of observation history: key -> (batch_size, history_length, ...)
        """
        return {key: jnp.stack(list(self.buffers[key]), axis=1) for key in self.buffers}
