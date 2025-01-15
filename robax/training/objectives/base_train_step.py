"""Common interface for all train steps while enforcing functional programming."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import attrs
import flax.linen as nn
import jax
import optax

from robax.utils.observation import Observation  # type: ignore

DEFAULT_PMAP_AXIS_NAME = "batch"


@attrs.define(frozen=True)
class BaseTrainStep(ABC):
    """Base train step class.

    Useage: override the get_loss method and the __call__ method. Within the __call__ method,
    call the compute_new_params method to get the new parameters, loss, and gradients.s
    """

    model: nn.Module = attrs.field()
    """The model to use for the train step."""
    optimizer: optax.GradientTransformation = attrs.field()
    """The optimizer to use for the train step."""
    unbatched_prediction_shape: Tuple[int, ...] = attrs.field()
    """Shape of the action to generate [A, a]."""
    do_pmap: bool = attrs.field(default=False)
    """Whether to parallelize the train step accross the batch axis via pmap."""
    pmap_axis_name: Optional[str] = attrs.field(default=DEFAULT_PMAP_AXIS_NAME)
    """The axis name to use for pmap."""

    ####################
    # Abstract methods #
    ####################

    @abstractmethod
    def get_loss(
        self,
        params: Dict[str, Any],
        observation: Observation,
        target: jax.Array,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the loss for the train step."""
        pass

    @abstractmethod
    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Computes the additional inputs for the train step."""
        pass

    ##################
    # Helper methods #
    ##################

    def compute_new_params(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        observation: Observation,
        target: jax.Array,
        **additional_inputs: jax.Array,
    ) -> Tuple[Dict[str, Any], optax.OptState, jax.Array, jax.Array]:
        """Computes the new parameters for the train step."""
        loss, grads = jax.value_and_grad(self.get_loss)(
            params, observation, target, **additional_inputs
        )
        if self.do_pmap:
            # If the call is pmap'd, we need to average the gradients and loss across the batch axis
            grads = jax.lax.pmean(grads, axis_name=self.pmap_axis_name)  # type: ignore
            loss = jax.lax.pmean(loss, axis_name=self.pmap_axis_name)  # type: ignore

        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)  # type: ignore
        return params, opt_state, loss, grads

    def __call__(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        observation: Observation,
        target: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any], optax.OptState, jax.Array, jax.Array]:
        """Call the train step and returns (prng_key, params, opt_state, loss, grads).

        Args:
            prng_key: The PRNG key to use for the train step.
            params: The parameters to use for the train step.
            opt_state: The optimizer state to use for the train step.
            observation: The observation to use for the train step.
            target: The target to use for the train step.
        Returns:
            prng_key: The PRNG key to use for the train step.
            params: The new parameters for the train step.
            opt_state: The new optimizer state for the train step.
            loss: The loss for the train step.
            grads: The gradients for the train step.
        """
        prng_key, additional_inputs = self.get_additional_inputs(prng_key, observation)
        if self.do_pmap:
            params, opt_state, loss, grads = jax.pmap(
                self.compute_new_params,
                axis_name=self.pmap_axis_name,
            )(params, opt_state, observation, target, **additional_inputs)
        else:
            params, opt_state, loss, grads = self.compute_new_params(
                params, opt_state, observation, target, **additional_inputs
            )

        return prng_key, params, opt_state, loss, grads
