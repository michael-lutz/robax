from abc import ABC, abstractmethod
from typing import Any, Dict

from robax.utils.observation import Observation


class BaseTrainObsTransform(ABC):
    """A class that contains a set of transformations to apply to a dataset."""

    @staticmethod
    @abstractmethod
    def format_obs_cpu(horizon: Dict[str, Any]) -> Dict[str, Any]:
        """Format into obs-like dictionary of numpy arrays using CPU-only operations."""
        pass

    @staticmethod
    @abstractmethod
    def transform_obs_gpu(horizon: Observation) -> Observation:
        """Apply the transformation to the horizon."""
        pass
