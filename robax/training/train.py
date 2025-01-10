"""Train the model"""

import os
import pickle
import jax
import jax.numpy as jnp
import optax
import wandb

from robax.training.data_utils.jax_dataloader import NaiveDataLoader
from robax.training.data_utils.jax_lerobot_dataset import JaxLeRobotDataset
from robax.training.model_from_config import get_model
from robax.training.objectives.flow_matching_action import train_step

checkpoint_dir = "../checkpoints/"

# Configuration for model selection
config = {
    "model_name": "PiZero",
    "model_arguments": {
        "vit_variant": "S/16",
        "llm_vocab_size": 0,  # shouldn't be used anyways...
        "gemma_mlp_dim": 2048,
        "gemma_embed_dim": 512,
        "action_expert_mlp_dim": 1024,
        "action_expert_embed_dim": 256,
        "depth": 12,
        "num_heads": 6,
        "num_kv_heads": 1,
        "head_dim": 64,
        "dropout": 0.1,
    },
}

log_every_n_steps = 1
save_every_n_steps = 20

# Load dataset
prng_key = jax.random.PRNGKey(0)

delta_timestamps = {
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.2, -0.1, 0.0],  # 3 for acceleration
    "action": [0.1, 0.2, 0.3, 0.4, 0.5],
}

dataset = JaxLeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
dataloader = NaiveDataLoader(prng_key, dataset, batch_size=128)

# Initialize wandb
wandb.init(project="robax", config=config)


model = get_model(config)

# Optimizer setup
optimizer = optax.adam(learning_rate=1e-3)

# Initialize parameters and optimizer state
params = model.init(
    jax.random.PRNGKey(0),
    jnp.ones((128, 2, 96, 96, 3)),  # images
    jnp.empty((128, 0), dtype=jnp.int32),  # text
    jnp.ones((128, 3, 2)),  # proprio
    jnp.ones((128, 5, 2)),  # action
    jnp.ones((128,)),  # timesteps
)
opt_state = optimizer.init(params)

# Example training loop
prng_key = jax.random.PRNGKey(0)
for epoch in range(10):  # Number of epochs
    for i, batch in enumerate(dataloader):
        prng_key, params, opt_state, loss = train_step(
            prng_key=prng_key,
            params=params,
            opt_state=opt_state,
            images=batch["image"],
            text=batch["text"],
            proprio=batch["proprio"],
            action=batch["action"],
            model=model,
            optimizer=optimizer,
        )

        # Log the loss to wandb
        if i % log_every_n_steps == 0:
            wandb.log({"epoch": epoch, "loss": loss})
            print(f"Epoch {epoch}, Loss: {loss}")

        if i % save_every_n_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{i}.pkl")
            os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
            with open(checkpoint_path, "wb") as f:
                pickle.dump(params, f)

# Finish the wandb run
wandb.finish()
