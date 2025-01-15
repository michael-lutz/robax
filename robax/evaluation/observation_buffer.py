"""Basic FIFO buffer for storing observations during evaluation."""

from collections import deque
from typing import Deque, Dict, List

import jax.numpy as jnp

from robax.utils.observation import Observation, observation_from_dict


class ObservationBuffer:
    """Basic FIFO buffer that returns JAX array with observation history.

    TODO: For environments with variance in timestamps / observation frequency, use timestamp logic.
    """

    def __init__(self, delta_timestamps: Dict[str, List[float]], horizon_dim: int = 1):
        """Initialize the observation buffer.

        Args:
            delta_timestamps: A dictionary of observation sizes.
            horizon_dim: The observation or action horizon/history dimension. Should be consistent.
        """
        self.observation_sizes = {
            key: len(timestamps) for key, timestamps in delta_timestamps.items()
        }
        self.buffers: Dict[str, Deque[jnp.ndarray]] = {
            key: deque(maxlen=self.observation_sizes[key]) for key in self.observation_sizes
        }
        self.horizon_dim = horizon_dim

    def add(self, observation: Dict[str, jnp.ndarray]) -> None:
        """Add an observation to the buffer. Note that every observation should have a batch dim.

        Args:
            observation: A dictionary of observations.
        """
        for key, value in observation.items():
            self.buffers[key].append(value)

    def get_observation(self) -> Observation:
        """Get the observation history as a JAX array, padded with the oldest data if necessary.

        Returns:
            A dictionary of observation history: key -> (batch_size, horizon_length, ...)
        """
        padded_buffers = {}
        for key, buffer in self.buffers.items():
            buffer_list = list(buffer)
            if len(buffer_list) < self.observation_sizes[key]:
                repeat_count = self.observation_sizes[key] - len(buffer_list)
                buffer_list = [buffer_list[0]] * repeat_count + buffer_list
            padded_buffers[key] = jnp.stack(buffer_list, axis=self.horizon_dim)

        return observation_from_dict(padded_buffers)
