"""Train the model"""

import argparse
import os
import pickle
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore

import wandb
from robax.config.base_training_config import Config
from robax.training.common_transforms import from_state_and_env_state
from robax.training.data_utils.dataloader import DataLoader
from robax.utils.model_utils import (
    get_model,
    get_objective,
    load_config,
    save_checkpoint,
)
from robax.utils.observation import Observation


def initialize_training(
    prng_key: jax.Array,
    config: Config,
) -> Tuple[
    jax.Array,
    DataLoader,
    nn.Module,
    optax.GradientTransformation,
    Dict[str, Any],
    optax.OptState,
]:
    """Initialize model, optimizer, and data loader.

    Args:
        config: The configuration.

    Returns:
        prng_key: The PRNG key.
        dataloader: The data loader.
        model: The model.
        optimizer: The optimizer.
        params: The parameters.
        opt_state: The optimizer state.
    """
    prng_key, subkey = jax.random.split(prng_key)
    batch_size = config["data"]["batch_size"]
    dataloader = DataLoader(
        dataset_id=config["data"]["dataset_id"],
        prng_key=subkey,
        delta_timestamps=config["data"]["delta_timestamps"],
        batch_size=batch_size,
        num_workers=config["data"]["num_workers"],
        shuffle=True,
        transform=from_state_and_env_state,  # TODO: create way to specify in config...
    )
    unbatched_prediction_shape = (
        config["data"]["action_target_length"],
        config["data"]["action_feature_size"],
    )
    model = get_model(
        config["model"],
        unbatched_prediction_shape=unbatched_prediction_shape,
    )
    optimizer = optax.adam(learning_rate=1e-3)

    init_observation: Observation = {
        "images": None,
        "text": None,
        "proprio": jnp.ones(
            (batch_size, config["data"]["proprio_length"], config["data"]["proprio_feature_size"])
        ),  # proprio # TODO: make this configurable
        "action": None,  # TODO: make this configurable
    }
    init_noisy_action = jnp.ones(
        (
            batch_size,
            config["data"]["action_history_length"] + config["data"]["action_target_length"],
            config["data"]["action_feature_size"],
        )
    )
    init_timesteps = jnp.ones((batch_size,))  # timesteps

    batch_size = config["data"]["batch_size"]
    params = model.init(
        jax.random.PRNGKey(0),
        observation=init_observation,
        noisy_action=init_noisy_action,
        timesteps=init_timesteps,
    )
    assert isinstance(params, dict)
    opt_state = optimizer.init(params)
    return prng_key, dataloader, model, optimizer, params, opt_state


def log(epoch: int, loss: jax.Array) -> None:
    """Log the loss and save model checkpoints.

    Args:
        epoch: The current epoch number.
        loss: The loss value to log.
    """
    wandb.log({"epoch": epoch, "loss": loss.item()})
    print(f"Epoch {epoch}, Loss: {loss.item()}")


def train_model(config: Config, checkpoint_dir: str) -> None:
    """Train the model using the specified configuration.

    Args:
        config: The configuration for training, including model, data, and training parameters.
    """
    prng_key, dataloader, model, optimizer, params, opt_state = initialize_training(
        jax.random.PRNGKey(config["training"]["seed"]), config
    )
    train_step = get_objective(config["objective"])

    # splitting action into historical and target
    try:
        action_timesteps = np.array(config["data"]["delta_timestamps"]["action"])
        action_split_idx = np.where(action_timesteps == 0.0)[0][0]
    except:
        raise ValueError("0.0 must be in delta_timestamps in the action key")
    assert action_split_idx == config["data"]["action_history_length"]

    for epoch in range(config["training"]["epochs"]):
        dataloader.randomize(prng_key)
        prng_key, _ = jax.random.split(prng_key)
        for i, batch in enumerate(dataloader):
            action = batch["action"]
            assert action is not None
            historical_action = action[:, :action_split_idx]
            if historical_action.shape[1] == 0:
                batch["action"] = None
            else:
                batch["action"] = historical_action

            target_action = action[:, action_split_idx:]
            prng_key, params, opt_state, loss, grads = train_step(
                prng_key=prng_key,
                params=params,
                opt_state=opt_state,
                model=model,
                optimizer=optimizer,
                observation=batch,
                target=target_action,
                unbatched_prediction_shape=(
                    config["data"]["action_target_length"],
                    config["data"]["action_feature_size"],
                ),
            )
            if i % config["training"]["log_every_n_steps"] == 0:
                log(epoch, loss)
            if i % config["training"]["save_every_n_steps"] == 0:
                save_checkpoint(params, checkpoint_dir, epoch, i)


def main() -> None:
    """Main function to execute the training process.

    Loads the configuration, initializes the wandb project, and starts the training process.
    """
    parser = argparse.ArgumentParser(description="Train the model with a specified config file.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="~/robax/robax/config/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="~/checkpoints",
        help="Directory to save checkpoints.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    wandb.init(
        project=config["project_name"],
        name=config["experiment_name"],
        config=config,  # type: ignore
    )
    train_model(config, args.checkpoint_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
