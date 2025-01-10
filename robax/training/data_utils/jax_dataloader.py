"""Naive dataloader for JAX."""

import jax
import jax.numpy as jnp


class NaiveDataLoader:
    def __init__(self, prng_key, dataset, batch_size):
        self.prng_key = prng_key
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(dataset)

    def __iter__(self):
        self.prng_key, subkey = jax.random.split(self.prng_key)
        self.indices = jax.random.permutation(subkey, self.dataset_size)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.dataset_size)
        batch_indices = self.indices[self.current_idx : end_idx]
        self.current_idx = end_idx

        batch = {"image": [], "text": [], "proprio": [], "action": []}

        for idx in batch_indices:
            data = self.dataset[int(idx)]
            batch["image"].append(data["image"])
            batch["text"].append(data["text"])
            batch["proprio"].append(data["proprio"])
            batch["action"].append(data["action"])

        # Convert lists to jax.numpy arrays and add batch dimension
        batch = {key: jnp.stack(value) for key, value in batch.items()}

        return batch
