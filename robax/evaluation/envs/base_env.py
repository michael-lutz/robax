"""Base environment for evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, SupportsFloat, Tuple

import numpy as np
import numpy.typing as npt

from robax.utils.numpy_observation import NumpyObservation


class BaseEnv(ABC):
    """Base environment for evaluation."""

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Tuple[int, ...]]:
        """Get the observation space."""
        pass

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NumpyObservation, dict[str, Any]]:
        """Reset the environment."""
        pass

    @abstractmethod
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[NumpyObservation, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        pass


class BatchableEnv(BaseEnv):
    """Base environment for evaluation.

    NOTE: while not strictly enforced, batchable environments should NOT use Jax for simplicity
    of multiprocessing.
    """

    @abstractmethod
    def get_factory_fn(cls) -> Callable[[], "BaseEnv"]:
        """Create an environment from a config."""
        pass
