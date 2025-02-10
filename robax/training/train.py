"""Train the model"""

import argparse
import time
from typing import Any, Dict, Hashable, Mapping, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore
from flax.core.frozen_dict import FrozenDict

import wandb
from robax.config.base_training_config import Config
from robax.evaluation.batch_evaluation import BatchEvaluator
from robax.model.policy.base_policy import BasePolicy
from robax.objectives.base_inference_step import BaseInferenceStep
from robax.objectives.base_train_step import BaseTrainStep
from robax.training.data_utils.dataloader import DataLoader
from robax.utils.model_utils import (
    get_dataloader,
    get_evaluator,
    get_inference_step,
    get_model,
    get_train_step,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from robax.utils.observation import Observation
from robax.utils.param_utils import load_params


def create_optimizer(
    base_learning_rate: float, overrides: Dict[str, float]
) -> optax.GradientTransformation:
    """Create an optimizer with learning rates specified for the different model parameters"""
    optimizers: Dict[Hashable, optax.GradientTransformation] = {
        "default": optax.adam(learning_rate=base_learning_rate)
    }
    optimizers.update(
        {
            param_name: optax.adam(learning_rate=learning_rate)
            for param_name, learning_rate in overrides.items()
        }
    )

    def labeler(current_param: Dict[str, Any], label: str = "default") -> Dict[str, Any]:
        res = {}
        if isinstance(current_param, jnp.ndarray):
            return label
        for key, value in current_param.items():
            if key in overrides:
                res[key] = labeler(value, key)
            else:
                res[key] = labeler(value, label)
        return res

    return optax.multi_transform(transforms=optimizers, param_labels=labeler)


def initialize_training(
    prng_key: jax.Array,
    config: Config,
    resume_from_checkpoint_path: str | None = None,
    vit_pretrained_checkpoint_path: str | None = None,
) -> Tuple[
    jax.Array,
    DataLoader,
    BasePolicy,
    optax.GradientTransformation,
    Dict[str, Any],
    optax.OptState,
    BatchEvaluator,
    BaseTrainStep,
    BaseInferenceStep,
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
        evaluator: The evaluator.
        train_step: The train step.
        inference_step: The inference step.
    """
    assert not (
        vit_pretrained_checkpoint_path is not None and resume_from_checkpoint_path is not None
    ), "Only one of vit_pretrained_checkpoint_path or resume_from_checkpoint_path can be provided"
    prng_key, subkey = jax.random.split(prng_key)
    batch_size = config["data"]["batch_size"]
    dataloader = get_dataloader(config["data"], subkey, batch_size)
    unbatched_prediction_shape = (
        config["data"]["action_target_length"],
        config["data"]["action_feature_size"],
    )
    model = get_model(
        config["model"],
        unbatched_prediction_shape=unbatched_prediction_shape,
    )

    # create optimizer with learning rates specified for the different model parameters
    overrides = {"img": 1e-4}  # TODO: make this configurable
    optimizer = create_optimizer(1e-3, overrides)

    init_observation: Observation = {
        "images": (
            jnp.ones(
                (
                    batch_size,
                    config["data"]["image_length"],
                    224,  # TODO: need to cleanly set this...
                    224,
                    3,
                )
            )
            if config["data"]["image_length"] > 0
            else None
        ),
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
    if resume_from_checkpoint_path is not None:
        params: FrozenDict[str, Mapping[str, Any]] | Dict[str, Any] = load_checkpoint(
            resume_from_checkpoint_path
        )
    else:
        params = model.init(
            jax.random.PRNGKey(config["training"]["seed"]),
            observation=init_observation,
            noisy_action=init_noisy_action,
            timesteps=init_timesteps,
        )

        if vit_pretrained_checkpoint_path is not None:
            vit_params = load_params(vit_pretrained_checkpoint_path)
            assert isinstance(vit_params, dict), "Haven't implemented FrozenDict loading yet"
            vit_params["head"] = params["params"]["img"]["head"]
            params["params"]["img"] = vit_params

    assert isinstance(params, dict)
    opt_state = optimizer.init(params)

    evaluator = get_evaluator(config)
    train_step = get_train_step(config["objective"])
    inference_step = get_inference_step(config["objective"], unbatched_prediction_shape)

    return (
        prng_key,
        dataloader,
        model,
        optimizer,
        params,
        opt_state,
        evaluator,
        train_step,
        inference_step,
    )


def train_model(
    config: Config,
    checkpoint_dir: str,
    resume_from_checkpoint_path: str | None = None,
    vit_pretrained_checkpoint_path: str | None = None,
    debug: bool = False,
) -> None:
    """Train the model using the specified configuration.

    Args:
        config: The configuration for training, including model, data, and training parameters.
    """
    if not debug:
        wandb.init(
            project=config["project_name"],
            name=config["experiment_name"],
            config=config,  # type: ignore
        )

    (
        prng_key,
        dataloader,
        model,
        optimizer,
        params,
        opt_state,
        evaluator,
        train_step,
        inference_step,
    ) = initialize_training(
        jax.random.PRNGKey(config["training"]["seed"]),
        config,
        resume_from_checkpoint_path,
        vit_pretrained_checkpoint_path,
    )

    @jax.jit
    def train_step_jit(
        prng_key: jax.Array,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        observation: Observation,
        target: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any], optax.OptState, jax.Array, jax.Array]:
        prng_key, params, opt_state, loss, grads = train_step(
            prng_key=prng_key,
            params=params,
            opt_state=opt_state,
            model=model,
            optimizer=optimizer,
            observation=observation,
            target=target,
            debug=debug,
            unbatched_prediction_shape=(
                config["data"]["action_target_length"],
                config["data"]["action_feature_size"],
            ),
        )
        return prng_key, params, opt_state, loss, grads

    @jax.jit
    def inference_step_jit(
        prng_key: jax.Array, params: Dict[str, Any], observation: Observation
    ) -> Tuple[jax.Array, jax.Array]:
        return inference_step.generate_action(prng_key, params, model, observation)

    # splitting action into historical and target
    action_split_idx = config["data"]["action_history_length"]
    num_batches_per_epoch = len(dataloader)
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
            starting_time = time.time()
            prng_key, params, opt_state, loss, _ = train_step_jit(
                prng_key=prng_key,
                params=params,
                opt_state=opt_state,
                observation=batch,
                target=target_action,
            )
            end_time = time.time()
            print(f"Time taken for train step: {end_time - starting_time}")
            total_steps = i + num_batches_per_epoch * epoch

            starting_time = time.time()
            if total_steps % config["training"]["log_every_n_steps"] == 0:
                if not debug:
                    wandb.log({"Epoch": epoch, "Loss": loss.item(), "Step": total_steps})
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            end_time = time.time()
            print(f"Time taken for logging: {end_time - starting_time}")
            if total_steps % config["training"]["save_every_n_steps"] == 0:
                save_checkpoint(params, checkpoint_dir, epoch, i)
            if total_steps % config["training"]["eval_every_n_steps"] == 0:
                prng_key, average_reward = evaluator.batch_rollout(
                    prng_key, params, inference_step_jit
                )
                print(f"Average reward at epoch {epoch}, step {i}: {average_reward}")
                if not debug:
                    wandb.log({"Eval": average_reward.item(), "Step": total_steps})

    if not debug:
        wandb.finish()


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
    parser.add_argument(
        "--resume_from_checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to resume from.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to use debug mode.",
    )
    parser.add_argument(
        "--vit_pretrained_checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to resume from.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    train_model(
        config,
        args.checkpoint_dir,
        args.resume_from_checkpoint_path,
        args.vit_pretrained_checkpoint_path,
        args.debug,
    )


if __name__ == "__main__":
    main()
