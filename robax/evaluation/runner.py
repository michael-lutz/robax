"""Runs the evaluation loop."""

from typing import Any, Dict

import gymnasium as gym
import jax
import jax.numpy as jnp

from robax.evaluation.observation_buffer import BatchedObservationBuffer


def batch_rollout(
    model: Any,
    env: gym.vector.VectorEnv,
) -> None:
    """Runs the evaluation loop."""

    observation_buffer = BatchedObservationBuffer(
        env.observation_space,
        env.action_space,
        history_length=10,
    )
