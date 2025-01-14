"""Base training config."""

from typing import Any, Dict, List, TypedDict


class DataConfig(TypedDict):
    repo_id: str
    delta_timestamps: Dict[str, List[float]]
    num_workers: int
    batch_size: int


class TrainingConfig(TypedDict):
    learning_rate: float  # TODO: add more optimizer hyperparameters
    epochs: int
    log_every_n_steps: int
    save_every_n_steps: int
    checkpoint_dir: str


class ObjectiveConfig(TypedDict):
    name: str
    args: Dict[str, Any]  # bespoke to each objective


class ModelConfig(TypedDict):
    name: str
    args: Dict[str, Any]  # bespoke to each model


class Config(TypedDict):
    experiment_name: str
    project_name: str
    data: DataConfig
    training: TrainingConfig
    objective: ObjectiveConfig
    model: ModelConfig
