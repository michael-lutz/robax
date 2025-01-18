"""Buffer for storing actions during inference."""

from typing import Sequence

import jax


class ActionBuffer:
    """Buffer for storing actions during inference."""

    def __init__(
        self,
        horizon_dim: int,
        action_inference_range: Sequence[int],
    ):
        """Initialize the trajectory buffer.

        Args:
            horizon_dim: dimension of the horizon
            action_inference_range: range of actions to generate
        """
        self.action_idx = action_inference_range[0]
        self.horizon_dim = horizon_dim
        self.trajectory: jax.Array | None = None
        self.action_inference_range = action_inference_range

    def is_empty(self) -> bool:
        """Check if the trajectory buffer is empty.

        Returns:
            True if the buffer is empty, False otherwise
        """
        if self.trajectory is None:
            return True

        return self.action_idx >= self.action_inference_range[1]

    def update_trajectorys(self, trajectory: jax.Array) -> None:
        """Update the trajectory buffer.

        Args:
            trajectory: trajectory to update the buffer with
        """
        self.trajectory = trajectory
        self.action_idx = self.action_inference_range[0]

    def pop_action(self) -> jax.Array:
        """Pop an action from the trajectory buffer.

        Returns:
            action: action to pop
        """
        if self.is_empty() or self.trajectory is None:  # mypy likes the second check...
            raise ValueError("Trajectory buffer has not been updated")
        action = self.trajectory[:, self.action_idx, :]
        self.action_idx += 1
        return action
