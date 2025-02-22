"""Base training config."""

from typing import Any, Dict, List, TypedDict


class DataConfig(TypedDict):
    dataset_id: str
    proprio_length: int
    action_history_length: int
    action_target_length: int
    action_feature_size: int
    proprio_feature_size: int
    image_length: int
    text_length: int
    delta_timestamps: Dict[str, List[float]]
    num_workers: int
    batch_size: int


class TrainingConfig(TypedDict):
    learning_rate: float  # TODO: add more optimizer hyperparameters
    epochs: int  # TODO: phase out with larget training setups
    log_every_n_steps: int
    save_every_n_steps: int
    eval_every_n_steps: int
    seed: int


class ObjectiveConfig(TypedDict):
    name: str
    args: Dict[str, Any]  # bespoke to each objective


class ModelConfig(TypedDict):
    name: str
    args: Dict[str, Any]  # bespoke to each model


class EvaluationConfig(TypedDict):
    env_name: str
    num_envs: int
    episode_length: int
    action_inference_range: List[int]


class Config(TypedDict):
    experiment_name: str
    project_name: str
    data: DataConfig
    training: TrainingConfig
    objective: ObjectiveConfig
    model: ModelConfig
    evaluation: EvaluationConfig
