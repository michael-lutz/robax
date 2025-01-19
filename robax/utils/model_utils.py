"""Model instantiation based on config"""

import os
import pickle
from typing import Any, Callable, Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
import yaml

from robax.config.base_training_config import (
    Config,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    ObjectiveConfig,
)
from robax.evaluation.batch_evaluation import BatchEvaluator
from robax.model.policy.base_policy import BasePolicy
from robax.objectives.base_inference_step import BaseInferenceStep
from robax.objectives.base_train_step import BaseTrainStep
from robax.training.data_utils.dataloader import DataLoader


def get_model(config: ModelConfig, unbatched_prediction_shape: Tuple[int, int]) -> BasePolicy:
    """Barebones config-based model instantiation

    Args:
        config: The model config.
        unbatched_prediction_shape: The shape of the unbatched prediction.
    Returns:
        The model.
    """

    if config["name"] == "pi_zero":
        from robax.model.policy.pi_zero import PiZero

        return PiZero(**config["args"], unbatched_prediction_shape=unbatched_prediction_shape)
    elif config["name"] == "mlp_policy":
        from robax.model.policy.mlp_policy import MLPPolicy

        return MLPPolicy(**config["args"], unbatched_prediction_shape=unbatched_prediction_shape)
    else:
        raise ValueError("Unknown model name in config")


def get_train_step(config: ObjectiveConfig) -> BaseTrainStep:
    """Barebones config-based objective instantiation

    Args:
        config: The objective config.

    Returns:
        The objective.
    """
    if config["name"] == "mse":
        from robax.objectives.mse import MSEObjective

        return MSEObjective()
    elif config["name"] == "flow_matching":
        from robax.objectives.flow_matching import FlowMatchingTrainStep

        return FlowMatchingTrainStep(
            cutoff_value=config["args"]["cutoff_value"],
            beta_a=config["args"]["beta_a"],
            beta_b=config["args"]["beta_b"],
        )

    elif config["name"] == "diffusion":
        from robax.objectives.diffusion import DiffusionTrainStep

        return DiffusionTrainStep()

    else:
        raise ValueError("Unknown objective name in config")


def get_inference_step(
    config: ObjectiveConfig, unbatched_prediction_shape: Tuple[int, int]
) -> BaseInferenceStep:
    """Barebones config-based objective instantiation

    Args:
        config: The objective config.

    Returns:
        The objective.
    """
    if config["name"] == "mse":
        from robax.objectives.mse import MSEInferenceStep

        return MSEInferenceStep()
    elif config["name"] == "diffusion":
        from robax.objectives.diffusion import DiffusionInferenceStep

        return DiffusionInferenceStep(
            unbatched_prediction_shape=unbatched_prediction_shape,
            num_steps=config["args"]["num_steps"],
            beta_start=config["args"]["beta_start"],
            beta_end=config["args"]["beta_end"],
        )
    elif config["name"] == "flow_matching":
        from robax.objectives.flow_matching import FlowMatchingInferenceStep

        return FlowMatchingInferenceStep(
            unbatched_prediction_shape=unbatched_prediction_shape,
            num_steps=config["args"]["num_steps"],
        )
    else:
        raise ValueError("Unknown objective name in config")


def get_dataloader(config: DataConfig, subkey: jnp.ndarray, batch_size: int) -> DataLoader:
    """Barebones config-based dataloader instantiation

    Args:
        config: The dataloader config.

    Returns:
        The dataloader.
    """
    if config["dataset_id"] == "push_t_keypoints":
        from robax.training.data_utils.obs_transforms.pusht_keypoints_transform import (
            PushTKeypointsTransform,
        )

        transform = PushTKeypointsTransform
    else:
        raise ValueError("Unknown dataset_id in config")

    dataloader = DataLoader(
        dataset_id=config["dataset_id"],
        prng_key=subkey,
        delta_timestamps=config["delta_timestamps"],
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=True,
        transform=transform,  # type: ignore
    )

    return dataloader


def get_evaluator(
    config: EvaluationConfig, unbatched_prediction_shape: Tuple[int, int]
) -> BatchEvaluator:
    """Barebones config-based evaluator instantiation

    Args:
        config: The evaluator config.

    Returns:
        The evaluator.
    """
    if config["env_name"] == "pusht":
        from robax.evaluation.envs.pusht_keypoints_env import PushTKeypointsEvalEnv

        create_env_fn = PushTKeypointsEvalEnv.get_factory_fn()

    else:
        raise ValueError("Unknown dataset_id in config")

    return BatchEvaluator(
        inference_step=get_inference_step(config["inference_step"], unbatched_prediction_shape),
        create_env_fn=create_env_fn,
        num_envs=config["num_envs"],
        observation_sizes=config["observation_sizes"],
        episode_length=config["episode_length"],
        action_inference_range=config["action_inference_range"],
    )


def load_config(path: str) -> Config:
    """Load config from yaml file.

    Args:
        path: Path to the yaml file.

    Returns:
        Config: The config.
    """
    with open(path, "r") as file:
        config: Config = yaml.safe_load(file)

    assert config["data"]["action_history_length"] + config["data"]["action_target_length"] == len(
        config["data"]["delta_timestamps"]["action"]
    )

    return config


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint from pkl file.

    Args:
        path: Path to the pkl file.

    Returns:
        Dict[str, Any]: The checkpoint.
    """
    with open(path, "rb") as file:
        checkpoint: Dict[str, Any] = pickle.load(file)

    return checkpoint


def save_checkpoint(params: Dict[str, Any], checkpoint_dir: str, epoch: int, i: int) -> None:
    """Save the model parameters to a checkpoint file.

    Args:
        params: The model parameters to save.
        checkpoint_dir: The directory where checkpoints will be saved.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{i}.pkl")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
    with open(checkpoint_path, "wb") as f:
        pickle.dump(params, f)
