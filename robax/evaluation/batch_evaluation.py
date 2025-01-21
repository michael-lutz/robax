"""Runs the evaluation loop."""

from typing import Any, Callable, Dict, List, Tuple

import attrs
import jax
import jax.numpy as jnp
import numpy as np

from robax.evaluation.action_buffer import ActionBuffer
from robax.evaluation.envs.base_env import BaseEnv
from robax.evaluation.envs.batched_env import BatchedEnv
from robax.evaluation.observation_buffer import ObservationBuffer
from robax.utils.observation import Observation, observation_from_numpy_observation


@attrs.define(frozen=True)
class BatchEvaluator:
    """Batch evaluator for a single environment."""

    create_env_fn: Callable[[], BaseEnv]
    num_envs: int
    observation_sizes: Dict[str, int | None]
    episode_length: int
    action_inference_range: List[int]

    def batch_rollout(
        self,
        prng: jax.Array,
        params: Dict[str, Any],
        generate_action_fn: Callable[
            [jax.Array, Dict[str, Any], Observation], Tuple[jax.Array, jax.Array]
        ],
    ) -> Tuple[jax.Array, jax.Array]:
        """Runs the evaluation loop and returns the average reward.

        Args:
            prng: [B] PRNG key
            params: model parameters
            model: model

        Returns:
            prng, average reward
        """

        batched_env = BatchedEnv(
            [self.create_env_fn() for _ in range(self.num_envs)], num_workers=1
        )

        observation_buffer = ObservationBuffer(
            observation_sizes=self.observation_sizes,
            horizon_dim=1,
        )

        trajectory_buffer = ActionBuffer(
            horizon_dim=1,
            action_inference_range=self.action_inference_range,
        )

        env_obs, _ = batched_env.reset()
        obs = observation_from_numpy_observation(env_obs)
        observation_buffer.add(obs)

        batch_rewards = jnp.zeros(self.num_envs)
        for _ in range(self.episode_length):
            observation_batch = observation_buffer.get_observation()

            prng, trajectory = generate_action_fn(prng, params, observation_batch)
            if trajectory_buffer.is_empty():
                trajectory_buffer.update_trajectorys(trajectory)

            action = trajectory_buffer.pop_action()
            env_obs, env_rewards, _, _, _ = batched_env.step(np.array(action))
            obs = observation_from_numpy_observation(env_obs)
            observation_buffer.add(obs)
            batch_rewards += jnp.array(env_rewards) / self.episode_length

        average_reward = jnp.mean(batch_rewards)

        return prng, average_reward
