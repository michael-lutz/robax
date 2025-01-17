import math
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterator, List

import jax.numpy as jnp
import jax.random
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from numpy.typing import NDArray
from PIL.PngImagePlugin import PngImageFile

from robax.utils.observation import Observation, observation_from_dict

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
    offset_timestamps: NDArray[np.float32],
    average_timestamp_length: float,
    tolerance: float = 1e-3,
) -> Any:
    """Given an index, get the row indices for the horizon.

    Edge cases:
        1) idx is near the start of an episode: replace with the first row in the episode
        2) idx is near the end of an episode: replace with the last row in the episode
        3) idx is near the start of the dataset: replace with the first row in the dataset
        4) idx is near the end of the dataset: replace with the last row in the dataset
        5) timestamps don't line up: forward fill based on the mask

    Args:
        ds: The dataset.
        idx: The index of the row we'd like to get data for.
        offset_timestamps: The array of float offsets we want (like [-0.1, 0.0, 0.1, ...]).
        average_timestamp_length: The average timestamp length (e.g., 0.1).
        tolerance: Tolerance for matching floating timestamps.

    Returns:
        An array of shape [horizon_length, ...]
    """
    cur_row = ds[idx]
    cur_timestamp = cur_row[TIMESTAMP]
    cur_horizon_idx = np.where(offset_timestamps == 0.0)[0][0]  # TODO: deal with no 0.0 case...
    expected_timestamps = np.clip(offset_timestamps + cur_timestamp, a_min=0.0, a_max=None)
    expected_idxs = np.round(offset_timestamps / average_timestamp_length).astype(int) + idx
    expected_idxs = np.clip(a=expected_idxs, a_min=0, a_max=len(ds) - 1)  # handles 3) and 4)

    data = ds[expected_idxs]

    timestamps = data[TIMESTAMP]

    mask = np.abs(timestamps - expected_timestamps) < tolerance  # handles 1), 2), and 5)
    assert mask[cur_horizon_idx], "This should never happen... (famous last words)"

    # Fill backwards from the current index
    for i in range(cur_horizon_idx - 1, -1, -1):
        if not mask[i]:
            data[feature_name][i] = data[feature_name][i + 1]

    # Forward fill
    for i in range(cur_horizon_idx + 1, len(mask)):
        if not mask[i]:
            data[feature_name][i] = data[feature_name][i - 1]

    return data[feature_name]


def get_horizon_helper(
    idx: int,
    ds: Dataset,
    delta_timestamps: Dict[str, NDArray[np.float32]],
    average_timestamp_length: float,
    tolerance: float = 1e-3,
    transform: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Given an index, get the entire horizon of images, states, and actions.

    Args:
        idx: The index of the row we'd like to get data for.
        ds: The dataset.
        delta_timestamps: A dict describing which offsets we want for each field.
        average_timestamp_length: The typical difference in timestamps (e.g., 0.1).
        tolerance: Tolerance for matching floating timestamps.
        transform: A function to transform the dataset.
    Returns:
        A dictionary of horizons.
    """
    horizon = {
        feature_name: get_horizon_feature(
            ds,
            idx,
            feature_name,
            delta_timestamps[feature_name],
            average_timestamp_length,
            tolerance,
        )
        for feature_name in delta_timestamps
    }

    if transform:
        horizon = transform(horizon)

    return horizon


def get_horizon_factory(
    ds: Dataset,
    delta_timestamps: Dict[str, NDArray[np.float32]],
    average_timestamp_length: float,
    tolerance: float = 1e-3,
    transform: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Callable[[int], Dict[str, Any]]:
    return partial(
        get_horizon_helper,
        ds=ds,
        delta_timestamps=delta_timestamps,
        average_timestamp_length=average_timestamp_length,
        tolerance=tolerance,
        transform=transform,
    )


def aggregate_horizons(horizons: List[Dict[str, jnp.ndarray]]) -> Observation:
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

    return observation_from_dict(batched_dict)  # type: ignore


class DataLoader:
    """High-performance DataLoader that returns batched jnp arrays of images, states, and actions.

    Attributes:
        dataset_id: str,
        delta_timestamps: Dict[str, List[float]],
        average_timestamp_length: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        tolerance: float = 1e-3,
        num_workers: int = 1,
    ):
        tolerance: Tolerance for matching floating timestamps.
        num_workers: If > 1, decodes images in parallel using a multiprocessing pool.
    """

    def __init__(
        self,
        prng_key: jnp.ndarray,
        dataset_id: str,
        delta_timestamps: Dict[str, List[float]],
        *,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        average_timestamp_length: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        tolerance: float = 1e-3,
        num_workers: int = 1,
    ):
        """Constructor.

        Args:
            prng_key: A JAX PRNG key.
            dataset_id: The id of the dataset to load or path to a local dataset.
            delta_timestamps: e.g.
                {
                    "observation.image": [-0.1, 0.0],
                    "observation.state": [-0.1, 0.0],
                    "action": [-0.1, 0.0, 0.1, ...],
                }
            transform: A function to transform the dataset.
            average_timestamp_length: The nominal delta between timestamps (0.1).
            batch_size: Number of samples per iteration.
            shuffle: Whether to shuffle the dataset indices before each epoch.
            tolerance: Tolerance used to match float timestamps.
            num_workers: Number of parallel workers to decode images (1=single-thread).
        """
        self.dataset = _load_dataset(dataset_id)
        self.transform = transform
        self.average_timestamp_length = average_timestamp_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tolerance = tolerance
        self.num_workers = num_workers

        self.delta_timestamps = {}
        for feature_name, offsets in delta_timestamps.items():
            self.delta_timestamps[feature_name] = np.array(offsets)

        self.num_rows = len(self.dataset)
        self.all_indices = jnp.arange(self.num_rows)
        self.randomize(prng_key)

        self.iter_ptr = 0

    def randomize(self, prng_key: jnp.ndarray) -> None:
        """Randomize the dataset indices."""
        if self.shuffle:
            self.all_indices = jax.random.permutation(prng_key, self.all_indices)

    def __len__(self) -> int:
        """Number of batches per 'epoch' if we consume all indices once."""
        return math.ceil(self.num_rows / self.batch_size)

    def __iter__(self) -> Iterator[Observation]:
        """Yield batches of (images, states, actions)."""
        self.iter_ptr = 0
        return self

    def __next__(self) -> Observation:
        """Get the next batch or raise StopIteration."""
        if self.iter_ptr >= self.num_rows:
            raise StopIteration

        # Sample next `self.batch_size` indices
        batch_end = min(self.iter_ptr + self.batch_size, self.num_rows)
        indices = self.all_indices[self.iter_ptr : batch_end].tolist()
        self.iter_ptr = batch_end

        get_horizon = get_horizon_factory(
            ds=self.dataset,
            delta_timestamps=self.delta_timestamps,
            average_timestamp_length=self.average_timestamp_length,
            tolerance=self.tolerance,
            transform=self.transform,
        )

        if self.num_workers > 1:
            with Pool(self.num_workers) as pool:
                batch = pool.map(get_horizon, indices)
        else:
            batch = [get_horizon(idx) for idx in indices]

        return aggregate_horizons(batch)
