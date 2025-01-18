"""Runs the evaluation loop."""

from typing import Any, Callable, Dict, List, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from robax.evaluation.action_buffer import ActionBuffer
from robax.evaluation.envs.base_env import BaseEnv
from robax.evaluation.envs.batched_env import BatchedEnv
from robax.evaluation.observation_buffer import ObservationBuffer
from robax.model.policy.base_policy import BasePolicy
from robax.utils.observation import Observation, observation_from_numpy_observation


def batch_rollout(
    prng: jax.Array,
    params: Dict[str, Any],
    model: BasePolicy,
    create_env_fn: Callable[[], BaseEnv],
    num_envs: int,
    observation_sizes: Dict[str, int | None],
    episode_length: int,
    action_inference_range: List[int],
    **generate_action_kwargs: Any,
) -> jax.Array:
    """Runs the evaluation loop and returns the average reward.

    Args:
        prng: [B] PRNG key
        params: model parameters
        model: model
        create_env_fn: function to create environment
        num_envs: number of environments
        observation_sizes: observation sizes
        episode_length: episode length
        action_inference_range: range of actions to generate
        **generate_action_kwargs: additional kwargs for generate_action

    Returns:
        average reward
    """

    batched_env = BatchedEnv([create_env_fn() for _ in range(num_envs)], num_workers=1)

    @jax.jit
    def jitted_generate_action(
        prng: jax.Array, params: Dict[str, Any], observation: Observation
    ) -> Tuple[jax.Array, jax.Array]:
        prng_key, action = model.apply(
            params,
            method="generate_action",
            prng=prng,
            observation=observation,
            **generate_action_kwargs,
        )
        assert isinstance(action, jax.Array)
        return prng_key, action

    observation_buffer = ObservationBuffer(
        observation_sizes=observation_sizes,
        horizon_dim=1,
    )

    trajectory_buffer = ActionBuffer(
        horizon_dim=1,
        action_inference_range=action_inference_range,
    )

    env_obs, _ = batched_env.reset()
    obs = observation_from_numpy_observation(env_obs)
    observation_buffer.add(obs)

    batch_rewards = jnp.zeros(num_envs)
    for _ in range(episode_length):
        observation_batch = observation_buffer.get_observation()

        prng, trajectory = jitted_generate_action(prng, params, observation_batch)
        if trajectory_buffer.is_empty():
            trajectory_buffer.update_trajectorys(trajectory)

        action = trajectory_buffer.pop_action()
        env_obs, env_rewards, _, _, _ = batched_env.step(np.array(action))
        obs = observation_from_numpy_observation(env_obs)
        observation_buffer.add(obs)
        batch_rewards += jnp.array(env_rewards) / episode_length

    average_reward = jnp.mean(batch_rewards)

    return average_reward
