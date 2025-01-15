from typing import Any, Dict

import numpy as np


def from_state_and_env_state(horizon: Dict[str, Any]) -> Dict[str, Any]:
    """Create a proprioception feature from the state and environment state."""
    out = {}
    normalizer = np.array([512])[None, :]  # TODO: parametrize
    out["proprio"] = (
        np.concatenate(
            [horizon["observation.state"], horizon["observation.environment_state"]], axis=-1
        )
        / normalizer
    )
    out["action"] = horizon["action"] / normalizer
    return out
