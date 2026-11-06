"""Regression coverage for the keys-only ActivationCache representation."""

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache


@pytest.mark.parametrize(
    "keys",
    [
        ["hook_embed"],
        ["hook_embed", "blocks.0.hook_resid_pre", "blocks.1.attn.hook_pattern"],
    ],
)
def test_repr_includes_cache_key_names(keys):
    cache = ActivationCache({key: torch.zeros(1) for key in keys}, model=None)

    representation = repr(cache)

    for key in keys:
        assert key in representation


def test_repr_does_not_represent_cached_values():
    class ReprForbiddenTensor(torch.Tensor):
        def __repr__(self):
            raise AssertionError("ActivationCache must not represent cached values")

    cache = ActivationCache(
        {
            key: torch.zeros(1).as_subclass(ReprForbiddenTensor)
            for key in ["hook_embed", "blocks.0.hook_resid_pre"]
        },
        model=None,
    )

    representation = repr(cache)

    assert "hook_embed" in representation
    assert "blocks.0.hook_resid_pre" in representation
