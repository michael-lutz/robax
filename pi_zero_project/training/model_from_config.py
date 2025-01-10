"""Model instantiation based on config"""

from typing import Any, Dict


def get_model(config: Dict[str, Any]):
    """Barebones config-based model instantiation"""

    if config["model_name"] == "PiZero":
        from pi_zero_project.model.policy.pi_zero import PiZero

        return PiZero(**config["model_arguments"])
    else:
        raise ValueError("Unknown model name in config")
