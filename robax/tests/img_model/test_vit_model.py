"""Basic testing suite for the ViT model."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from pytest_snapshot.plugin import Snapshot

from robax.model.img_model.vit import Model


@pytest.fixture
def model():
    # Create a small variant of the model for testing
    return Model(num_classes=10, variant="mu/16")


def test_model_initialization(model: nn.Module) -> None:
    assert model is not None


def test_model_forward_pass(model: nn.Module, snapshot: Snapshot) -> None:
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input, train=False)
    output, output_dict = model.apply(params, dummy_input, train=False)
    pre_logits = output_dict["pre_logits"]
    assert isinstance(pre_logits, jnp.ndarray)

    snapshot.assert_match(pre_logits.tobytes(), "forward_pass_output")
    assert output.shape == (1, 10), "Output shape mismatch"


def test_model_consistency(model: nn.Module) -> None:
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input, train=False)
    output1, _ = model.apply(params, dummy_input, train=False)
    output2, _ = model.apply(params, dummy_input, train=False)

    assert jnp.allclose(output1, output2), "Outputs are not consistent"


if __name__ == "__main__":
    pytest.main()
