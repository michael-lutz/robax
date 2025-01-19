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
from robax.model.policy.base_policy import BasePolicy
from robax.objectives.base_inference_step import BaseInferenceStep
from robax.utils.observation import Observation, observation_from_numpy_observation


@attrs.define(frozen=True)
class BatchEvaluator:
    """Batch evaluator for a single environment."""

    create_env_fn: Callable[[], BaseEnv]
    num_envs: int
    observation_sizes: Dict[str, int | None]
    episode_length: int
    action_inference_range: List[int]
    inference_step: BaseInferenceStep

    def batch_rollout(
        self,
        prng: jax.Array,
        params: Dict[str, Any],
        model: BasePolicy,
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

        @jax.jit
        def jitted_generate_action(
            prng: jax.Array, params: Dict[str, Any], observation: Observation
        ) -> Tuple[jax.Array, jax.Array]:
            prng_key, action = self.inference_step.generate_action(
                prng_key=prng,
                params=params,
                model=model,
                observation=observation,
            )
            assert isinstance(action, jax.Array)
            return prng_key, action

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

            prng, trajectory = jitted_generate_action(prng, params, observation_batch)
            if trajectory_buffer.is_empty():
                trajectory_buffer.update_trajectorys(trajectory)

            action = trajectory_buffer.pop_action()
            env_obs, env_rewards, _, _, _ = batched_env.step(np.array(action))
            obs = observation_from_numpy_observation(env_obs)
            observation_buffer.add(obs)
            batch_rewards += jnp.array(env_rewards) / self.episode_length

        average_reward = jnp.mean(batch_rewards)

        return prng, average_reward
