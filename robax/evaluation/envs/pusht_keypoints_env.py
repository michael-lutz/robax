"""Pusht keypoints env for evaluation."""

from typing import Any, Callable, Dict, SupportsFloat, Tuple

import gym_pusht  # type: ignore
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from robax.evaluation.envs.base_env import BatchableEnv
from robax.utils.numpy_observation import NumpyObservation, numpy_observation_from_dict


class PushTKeypointsEvalEnv(BatchableEnv):
    """Pusht keypoints env for evaluation."""

    def __init__(self) -> None:
        """Initialize the environment."""
        self.underlying_env = gym.make(
            "gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="environment_state_agent_pos"
        )

    def transform_observation(self, observation: Dict[str, Any]) -> NumpyObservation:
        """Process the observation."""
        out = {
            "proprio": np.concatenate(
                [observation["agent_pos"], observation["environment_state"]], axis=-1
            )
            / 512,
        }

        return numpy_observation_from_dict(out)

    @property
    def observation_space(self) -> Dict[str, Tuple[int, ...]]:
        """Get the observation space."""
        # TODO: should this exist in the base class?
        agent_pos_size = self.underlying_env.observation_space["agent_pos"].shape[0]  # type: ignore
        environment_state_size = self.underlying_env.observation_space["environment_state"].shape[0]  # type: ignore
        proprio_size = agent_pos_size + environment_state_size
        return {"proprio": (proprio_size,)}

    def reset(self, **kwargs: Any) -> Tuple[NumpyObservation, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            The observation and info.
        """
        observation, info = self.underlying_env.reset(**kwargs)
        return self.transform_observation(observation), info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[NumpyObservation, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment.

        Args:
            action: The normalized action to take, ranging from 0 to 1.

        Returns:
            The observation, reward, terminated, truncated, and info.
        """
        assert isinstance(action, np.ndarray)
        action = action * 512
        observation, reward, terminated, truncated, info = self.underlying_env.step(action)
        return self.transform_observation(observation), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the environment."""
        render = self.underlying_env.render()
        assert isinstance(render, np.ndarray)
        return render

    @classmethod
    def get_factory_fn(cls) -> Callable[[], "PushTKeypointsEvalEnv"]:
        """Create an environment from a config."""
        return lambda: cls()
