"""Regression coverage for the keys-only ActivationCache representation."""

from io import StringIO

import pytest
import torch
from IPython.core.formatters import DisplayFormatter

from transformer_lens.ActivationCache import ActivationCache


def printed_text(cache):
    output = StringIO()
    print(cache, file=output)
    return output.getvalue()


def ipython_text(cache):
    data, _ = DisplayFormatter().format(cache)
    return data.get("text/plain", "")


@pytest.fixture(
    params=[repr, str, printed_text, ipython_text], ids=["repr", "str", "print", "ipython"]
)
def render(request):
    return request.param


@pytest.mark.parametrize(
    "keys",
    [
        ["hook_embed"],
        ["hook_embed", "blocks.0.hook_resid_pre", "blocks.1.attn.hook_pattern"],
    ],
)
def test_representation_includes_cache_key_names(keys, render):
    cache = ActivationCache({key: torch.zeros(1) for key in keys}, model=None)

    representation = render(cache)

    for key in keys:
        assert key in representation


def test_representation_does_not_represent_cached_values(render):
    repr_calls = []

    class ReprForbiddenTensor(torch.Tensor):
        def __repr__(self):
            repr_calls.append(id(self))
            raise AssertionError("ActivationCache must not represent cached values")

    cache = ActivationCache(
        {
            key: torch.zeros(1).as_subclass(ReprForbiddenTensor)
            for key in ["hook_embed", "blocks.0.hook_resid_pre"]
        },
        model=None,
    )

    representation = render(cache)

    # IPython can catch formatter exceptions, so also check attempted calls.
    assert repr_calls == []
    assert "hook_embed" in representation
    assert "blocks.0.hook_resid_pre" in representation
