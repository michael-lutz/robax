"""Basic FIFO buffer for storing observations during evaluation."""

from collections import deque
from typing import Deque, Dict

import jax.numpy as jnp

from robax.utils.observation import Observation, observation_from_dict


class ObservationBuffer:
    """Basic FIFO buffer that returns JAX array with observation history.

    TODO: For environments with variance in timestamps / observation frequency, use timestamp logic.
    """

    def __init__(self, observation_sizes: Dict[str, int], horizon_dim: int = 1):
        """Initialize the observation buffer.

        Args:
            observation_sizes: A dictionary of observation sizes.
            horizon_dim: The observation or action horizon/history dimension. Should be consistent.
        """
        self.observation_sizes = observation_sizes
        self.buffers: Dict[str, Deque[jnp.ndarray]] = {
            obs_name: deque(maxlen=obs_size)
            for obs_name, obs_size in self.observation_sizes.items()
            if obs_size is not None
        }
        self.horizon_dim = horizon_dim

    def add(self, observation: Dict[str, jnp.ndarray]) -> None:
        """Add an observation to the buffer. Note that every observation should have a batch dim.

        Args:
            observation: A dictionary of observations.
        """
        for key, value in observation.items():
            if key in self.buffers and value is not None:
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
