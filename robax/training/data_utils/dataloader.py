import math
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterator, List

import jax.numpy as jnp
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from PIL.PngImagePlugin import PngImageFile

TIMESTAMP = "timestamp"
EPISODE_IDX = "episode_index"


def _load_dataset(dataset_id: str) -> Dataset:
    """Load a dataset from the Hugging Face Hub, or a local dataset.

    Args:
        dataset_id: The id of the dataset to load or path to a local dataset.

    Returns:
        A Huggingface Dataset.
    """
    if dataset_id.endswith(".arrow"):
        dataset = Dataset.from_file(dataset_id, in_memory=False)
    else:
        dataset = load_dataset(dataset_id, split=None)
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            dataset = dataset["train"]
    return dataset  # type: ignore


def convert_to_jnp(feature: Any) -> jnp.ndarray:
    """Convert a feature to a jnp array.

    Args:
    feature: An object that can be converted into a jnp array.

    Returns:
    A jnp array (float32) with shape [horizon_length, ...]
    """
    if isinstance(feature, PngImageFile):
        np_img = np.array(feature, dtype=np.uint8)  # shape: (H, W, C)
        return jnp.array(np_img)
    elif isinstance(feature, np.ndarray) or isinstance(feature, list):
        return jnp.array(feature)
    else:
        raise ValueError(f"Unsupported feature type: {type(feature)}")


def get_horizon_feature(
    ds: Dataset,
    idx: int,
    feature_name: str,
    offset_timesteps: np.ndarray,
    average_timestep_length: float,
    tolerance: float = 1e-3,
) -> Any:
    """Given an index, get the row indices for the horizon.

    Edge cases:
        1) idx is near the start of an episode: replace with the first row in the episode
        2) idx is near the end of an episode: replace with the last row in the episode
            2a) Idx is near the end of the dataset: handle in the indexing phase
            2b) Idx is near the end of an episode: handle retroactively via a mask
        3) Timesteps don't line up: forward fill based on the mask

    Args:
        ds: The dataset.
        idx: The index of the row we'd like to get data for.
        offset_timesteps: The array of float offsets we want (like [-0.1, 0.0, 0.1, ...]).
        average_timestep_length: The average timestep length (e.g., 0.1).
        tolerance: Tolerance for matching floating timesteps.

    Returns:
        An array of shape [horizon_length, ...]
    """
    cur_row = ds[idx]
    cur_timestep = cur_row[TIMESTAMP]
    cur_horizon_idx = np.where(offset_timesteps == 0.0)[0][0]
    expected_timesteps = np.clip(offset_timesteps + cur_timestep, a_min=cur_timestep, a_max=None)
    expected_idxs = np.round(offset_timesteps / average_timestep_length).astype(int) + idx
    expected_idxs = np.clip(a=expected_idxs, a_min=idx, a_max=len(ds) - 1)  # 1) and 2a)

    data = ds[expected_idxs]

    # 2b and 3) Handles new episode & timesteps that don't line up
    timesteps = data[TIMESTAMP]
    mask = np.abs(timesteps - expected_timesteps) < tolerance
    if not mask[0]:
        warnings.warn(f"First row for {feature_name} is not within tolerance of {cur_timestep}.")
        # TODO: perhaps should skip this data if the timesteps are too off...
        data[feature_name][0] = data[feature_name][cur_horizon_idx]

    # Forward fill based on the mask
    for i in range(1, len(mask)):
        if not mask[i]:
            data[feature_name][i] = data[feature_name][i - 1]

    return data[feature_name]


def get_horizon_helper(
    idx: int,
    ds: Dataset,
    delta_timesteps: Dict[str, np.ndarray],
    average_timestep_length: float,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Given an index, get the entire horizon of images, states, and actions.

    Args:
        idx: The index of the row we'd like to get data for.
        ds: The dataset.
        delta_timesteps: A dict describing which offsets we want for each field.
        average_timestep_length: The typical difference in timesteps (e.g., 0.1).
        tolerance: Tolerance for matching floating timesteps.

    Returns:
        A dictionary of horizons.
    """
    return {
        feature_name: get_horizon_feature(
            ds, idx, feature_name, delta_timesteps[feature_name], average_timestep_length, tolerance
        )
        for feature_name in delta_timesteps
    }


def get_horizon_factory(
    ds: Dataset,
    delta_timesteps: Dict[str, np.ndarray],
    average_timestep_length: float,
    tolerance: float = 1e-3,
) -> Callable[[int], Dict[str, Any]]:
    return partial(
        get_horizon_helper,
        ds=ds,
        delta_timesteps=delta_timesteps,
        average_timestep_length=average_timestep_length,
        tolerance=tolerance,
    )


def aggregate_horizons(horizons: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    """Aggregate a list of horizons into a single jnp batch.

    Args:
        horizons: A list of horizons.

    Returns:
        A dictionary of batched horizons.
    """
    keys = horizons[0].keys()
    batched_dict = {}
    for key in keys:
        batched_dict[key] = jnp.stack([convert_to_jnp(h[key]) for h in horizons])
    return batched_dict


class HorizonBatchLoader:
    """High-performance DataLoader that returns batched jnp arrays of images, states, and actions.

    Attributes:
        dataset_id: str,
        delta_timesteps: Dict[str, List[float]],
        average_timestep_length: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        tolerance: float = 1e-3,
        num_workers: int = 1,
    ):
        tolerance: Tolerance for matching floating timesteps.
        num_workers: If > 1, decodes images in parallel using a multiprocessing pool.
    """

    def __init__(
        self,
        prng_key: jnp.ndarray,
        dataset_id: str,
        delta_timesteps: Dict[str, List[float]],
        *,
        average_timestep_length: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        tolerance: float = 1e-3,
        num_workers: int = 1,
    ):
        """Constructor.

        Args:
            prng_key: A JAX PRNG key.
            dataset_id: The id of the dataset to load or path to a local dataset.
            delta_timesteps: e.g.
                {
                    "observation.image": [-0.1, 0.0],
                    "observation.state": [-0.1, 0.0],
                    "action": [-0.1, 0.0, 0.1, ...],
                }
            average_timestep_length: The nominal delta between timesteps (0.1).
            batch_size: Number of samples per iteration.
            shuffle: Whether to shuffle the dataset indices before each epoch.
            tolerance: Tolerance used to match float timesteps.
            num_workers: Number of parallel workers to decode images (1=single-thread).
        """
        self.dataset = _load_dataset(dataset_id)
        self.average_timestep_length = average_timestep_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tolerance = tolerance
        self.num_workers = num_workers

        self.delta_timesteps = {}
        for feature_name, offsets in delta_timesteps.items():
            self.delta_timesteps[feature_name] = np.array(offsets)

        self.num_rows = len(self.dataset)
        self.all_indices = np.arange(self.num_rows)
        self.randomize(prng_key)

        self.iter_ptr = 0

    def randomize(self, prng_key: jnp.ndarray) -> None:
        """Randomize the dataset indices."""
        self.prng_key = prng_key
        if self.shuffle:
            np.random.shuffle(self.all_indices)

    def __len__(self) -> int:
        """Number of batches per 'epoch' if we consume all indices once."""
        return math.ceil(self.num_rows / self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Yield batches of (images, states, actions)."""
        self.iter_ptr = 0
        return self

    def __next__(self) -> Dict[str, jnp.ndarray]:
        """Get the next batch or raise StopIteration."""
        if self.iter_ptr >= self.num_rows:
            raise StopIteration

        # Sample next `self.batch_size` indices
        batch_end = min(self.iter_ptr + self.batch_size, self.num_rows)
        indices = self.all_indices[self.iter_ptr : batch_end].tolist()
        self.iter_ptr = batch_end

        get_horizon = get_horizon_factory(
            ds=self.dataset,
            delta_timesteps=self.delta_timesteps,
            average_timestep_length=self.average_timestep_length,
            tolerance=self.tolerance,
        )

        if self.num_workers > 1:
            # We'll decode images in parallel
            with Pool(self.num_workers) as pool:
                batch = pool.map(get_horizon, indices)
        else:
            # We'll decode images in serial
            batch = [get_horizon(idx) for idx in indices]

        return aggregate_horizons(batch)
