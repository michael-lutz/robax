"""Batched environment for evaluation."""

from typing import Any, Dict, Sequence, SupportsFloat, Tuple

import numpy as np
import numpy.typing as npt

from robax.evaluation.envs.base_env import BaseEnv
from robax.utils.numpy_observation import NumpyObservation, numpy_observation_from_dict


class BatchedEnv(BaseEnv):
    """Batched environment for evaluation."""

    def __init__(self, envs: Sequence[BaseEnv], num_workers: int) -> None:
        """Initialize the environment."""
        if num_workers > 1:
            raise NotImplementedError("Multi-worker environment batching is not supported yet")

        # TODO: assert the expected observation keys are the same for all environments

        self.envs = envs
        self.num_workers = num_workers

    @property
    def observation_space(self) -> Dict[str, Tuple[int, ...]]:
        """Get the observation space."""
        unbatched_observation_space = self.envs[0].observation_space
        return {
            key: (len(self.envs), *unbatched_observation_space[key])
            for key in unbatched_observation_space.keys()
        }

    def reset(self, **kwargs: Any) -> Tuple[NumpyObservation, Dict[str, Any]]:
        """Reset the environment."""
        observations, infos = zip(*[env.reset(**kwargs) for env in self.envs])

        # Adding observations to the batch dimension of each key
        batched_observation = {
            key: np.stack([obs[key] for obs in observations])
            for key in observations[0].keys()
            if observations[0][key] is not None
        }
        obs = numpy_observation_from_dict(batched_observation)

        return obs, infos

    def step(
        self, actions: npt.NDArray[np.float32]
    ) -> Tuple[NumpyObservation, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        assert actions.shape[0] == len(
            self.envs
        ), "Number of actions must match number of environments"

        observations, rewards, terminateds, truncateds, infos = zip(
            *[env.step(actions[i]) for i, env in enumerate(self.envs)]
        )

        # stacking the observations
        batched_observation = {
            key: np.stack([obs[key] for obs in observations])
            for key in observations[0].keys()
            if observations[0][key] is not None
        }
        obs = numpy_observation_from_dict(batched_observation)

        return obs, rewards, terminateds, truncateds, infos
