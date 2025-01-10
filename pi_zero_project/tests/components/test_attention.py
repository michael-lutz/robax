# test_attention.py

import pytest
import jax.numpy as jnp
from pi_zero_project.model.components.attention import (
    make_attn_mask,
    apply_attention,
)


# Test cases for make_attn_mask
@pytest.mark.parametrize(
    "input_mask, mask_ar, expected_shape",
    [
        (jnp.array([[True, True, False]]), jnp.array([[1, 0, 0]]), (1, 3, 3)),
        (jnp.array([[True, True, True]]), jnp.array([[0, 0, 0]]), (1, 3, 3)),
        (jnp.array([[True, False, True]]), jnp.array([[1, 1, 0]]), (1, 3, 3)),
        (jnp.array([[False, False, False]]), jnp.array([[0, 0, 0]]), (1, 3, 3)),
        (jnp.array([[True, True, True, True]]), jnp.array([[1, 1, 1, 0]]), (1, 4, 4)),
    ],
)
def test_make_attn_mask(input_mask, mask_ar, expected_shape):
    attn_mask = make_attn_mask(input_mask, mask_ar)
    assert attn_mask.shape == expected_shape
    assert attn_mask.dtype == jnp.bool_


def test_apply_attention_case1():
    q = jnp.ones((1, 3, 4, 16))
    k = jnp.ones((1, 3, 1, 16))
    v = jnp.ones((1, 3, 1, 16))
    attn_mask = jnp.ones((1, 1, 3, 3))
    attn_logits_softcap = None

    expected_output = jnp.ones((1, 3, 4, 16))
    output = apply_attention(q, k, v, attn_mask, attn_logits_softcap)
    assert jnp.allclose(output, expected_output), "Output does not match expected output for case 1"


def test_apply_attention_case2():
    q = jnp.array([1, 1, 1])[None, None, None, :]
    k = jnp.array([1, 1, 1])[None, None, None, :]
    v = jnp.array([1, 1, 1])[None, None, None, :]
    attn_mask = jnp.ones((1, 1, 1, 1))
    attn_logits_softcap = 0.5

    expected_output = jnp.ones((1, 1, 1, 3))
    output = apply_attention(q, k, v, attn_mask, attn_logits_softcap)
    assert jnp.allclose(output, expected_output), "Output does not match expected output for case 2"


def test_no_overlap_of_q_and_k():
    q = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[None, None, :, :]
    k = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[None, None, :, :]
    v = jnp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])[None, None, :, :]
    attn_mask = jnp.ones((1, 1, 1, 1))
    attn_logits_softcap = 0.5

    expected_output = jnp.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])[None, None, :, :]
    output = apply_attention(q, k, v, attn_mask, attn_logits_softcap)
    assert jnp.allclose(
        output, expected_output
    ), "Output does not match expected output for no overlap of q and k"


# TODO: Add more unit tests
