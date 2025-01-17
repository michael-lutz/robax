"""Common interface for all train steps while enforcing functional programming."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import attrs
import flax.linen as nn
import jax
import optax  # type: ignore

from robax.model.components.axes_names import BATCH_AXIS
from robax.utils.observation import Observation  # type: ignore

DEFAULT_PMAP_AXIS_NAME = BATCH_AXIS


@attrs.define(frozen=True)
class BaseTrainStep(ABC):
    """Base train step class.

    Useage: override the get_loss method and the __call__ method. Within the __call__ method,
    call the compute_new_params method to get the new parameters, loss, and gradients.s
    """

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
        model: nn.Module,
        observation: Observation,
        target: jax.Array,
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the loss for the train step."""
        pass

    @abstractmethod
    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
        unbatched_prediction_shape: Tuple[int, int],
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
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        observation: Observation,
        target: jax.Array,
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> Tuple[Dict[str, Any], optax.OptState, jax.Array, jax.Array]:
        """Computes the new parameters for the train step.

        Args:
            params: The parameters to use for the train step.
            opt_state: The optimizer state to use for the train step.
            model: The model to use for the train step.
            optimizer: The optimizer to use for the train step.
            observation: The observation to use for the train step.
            target: The target to use for the train step.
            **additional_inputs: Additional inputs to use for the train step.

        Returns:
            params: The new parameters for the train step.
            opt_state: The new optimizer state for the train step.
            loss: The loss for the train step.
            grads: The gradients for the train step.
        """
        loss, grads = jax.value_and_grad(self.get_loss)(
            params,
            model=model,
            observation=observation,
            target=target,
            debug=debug,
            **additional_inputs,
        )
        if self.do_pmap:
            # If the call is pmap'd, we need to average the gradients and loss across the batch axis
            grads = jax.lax.pmean(grads, axis_name=self.pmap_axis_name)  # type: ignore
            loss = jax.lax.pmean(loss, axis_name=self.pmap_axis_name)  # type: ignore

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)  # type: ignore
        return params, opt_state, loss, grads

    def __call__(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        observation: Observation,
        target: jax.Array,
        unbatched_prediction_shape: Tuple[int, int],
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any], optax.OptState, jax.Array, jax.Array]:
        """Call the train step and returns (prng_key, params, opt_state, loss, grads).

        Args:
            prng_key: The PRNG key to use for the train step.
            params: The parameters to use for the train step.
            opt_state: The optimizer state to use for the train step.
            model: The model to use for the train step.
            optimizer: The optimizer to use for the train step.
            observation: The observation to use for the train step.
            target: The target to use for the train step.
            unbatched_prediction_shape: The shape of the unbatched prediction.
            debug: Whether to use debug mode.
            **additional_inputs: Additional inputs to use for the train step.

        Returns:
            prng_key: The PRNG key to use for the train step.
            params: The new parameters for the train step.
            opt_state: The new optimizer state for the train step.
            loss: The loss for the train step.
            grads: The gradients for the train step.
        """
        prng_key, additional_inputs = self.get_additional_inputs(
            prng_key=prng_key,
            observation=observation,
            unbatched_prediction_shape=unbatched_prediction_shape,
        )
        if self.do_pmap:
            params, opt_state, loss, grads = jax.pmap(
                self.compute_new_params,
                axis_name=self.pmap_axis_name,
            )(params, opt_state, model, optimizer, observation, target, debug, **additional_inputs)
        else:
            params, opt_state, loss, grads = self.compute_new_params(
                params=params,
                opt_state=opt_state,
                model=model,
                optimizer=optimizer,
                observation=observation,
                target=target,
                debug=debug,
                **additional_inputs,
            )

        return prng_key, params, opt_state, loss, grads
