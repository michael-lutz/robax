from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainObsTransform(ABC):
    """A class that contains a set of transformations to apply to a dataset."""

    @staticmethod
    @abstractmethod
    def __call__(horizon: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transformation to the horizon."""
        pass
