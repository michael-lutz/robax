"""Basic testing suite for the Gemma model."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.numpy as jnp
from pi_zero_project.model.vlm.llm_model.gemma import Model
from pytest_snapshot.plugin import Snapshot


@pytest.fixture
def model():
    # Create a small variant of the model for testing
    return Model(
        variant="gemma_2b",
        width=32,
        depth=1,
        mlp_dim=16,
        num_heads=1,
        num_kv_heads=1,
        head_dim=16,
        norm_eps=1e-6,
        vocab_size=100,
        scan=True,
        remat_policy="nothing_saveable",
    )


@pytest.mark.skip(reason="Skipping since Gemma isn't directly used right now")
def test_model_initialization(model: Model) -> None:
    assert model is not None


@pytest.mark.skip(reason="Skipping model forward pass since Gemma isn't directly used right now")
def test_model_forward_pass(model: Model, snapshot: Snapshot) -> None:
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, tokens)
    logits, _ = model.apply(params, tokens)

    snapshot.assert_match(logits.tobytes(), "forward_pass_logits")
    assert logits.shape == (1, 10, model.vocab_size), "Logits shape mismatch"


@pytest.mark.skip(
    reason="Skipping model consistency test since Gemma isn't directly used right now"
)
def test_model_consistency(model: Model) -> None:
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, tokens)
    logits1, _ = model.apply(params, tokens)
    logits2, _ = model.apply(params, tokens)

    assert jnp.allclose(logits1, logits2), "Logits are not consistent"


if __name__ == "__main__":
    pytest.main()
