"""Runs the evaluation loop."""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp

from robax.evaluation.observation_buffer import ObservationBuffer


def batch_rollout(
    prng: jax.Array,
    params: Any,
    model: Any,
    env: gym.vector.VectorEnv,
    delta_timestamps: Dict[str, List[float]],
    episode_length: int,
    action_shape_to_generate: Tuple[int, ...],
    **generate_action_kwargs: Any,
) -> int:
    """Runs the evaluation loop and returns the number of successful episodes.

    Args:
        prng: [B] PRNG key
        params: model parameters
        model: model
        env: environment
        delta_timestamps: delta timestamps
        episode_length: episode length
        action_shape_to_generate: action shape to generate
        **generate_action_kwargs: additional kwargs for generate_action

    Returns:
        number of successful episodes
    """

    @jax.jit
    def jitted_generate_action(prng, params, observation):
        return model.apply(
            prng=prng,
            params=params,
            observation=observation,
            method="generate_action",
            action_shape_to_generate=action_shape_to_generate,
            **generate_action_kwargs,
        )

    observation_buffer = ObservationBuffer(
        delta_timestamps=delta_timestamps,
        horizon_dim=1,
    )
    observations, infos = env.reset()
    observation_buffer.add(observations)
    success_count = 0
    for _ in range(episode_length):
        observation_batch = observation_buffer.get_observation()

        action = jitted_generate_action(prng, params, observation_batch)

        observations, _, terminateds, _, _ = env.step(action)
        observation_buffer.add(observations)
        success_count += int(jnp.sum(terminateds))

    return success_count
