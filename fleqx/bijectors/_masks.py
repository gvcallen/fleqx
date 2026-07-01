"""Masks used to build autoregressive neural networks."""

from __future__ import annotations

import operator

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def rank_based_mask(
    in_ranks: Int[Array, " a"], out_ranks: Int[Array, " b"], *, eq: bool = False
) -> Bool[Array, "b a"]:
    """A mask with entries `out_ranks > in_ranks` (or `>=`, if `eq`).

    Used to build a MADE-style masked MLP (Germain et al., 2015): zeroing a weight
    matrix with this mask ensures the layer's output at rank `r` only depends on
    inputs with a strictly lower rank, which is what makes the resulting network
    autoregressive.

    **Returns:**

    - A boolean array of shape `(len(out_ranks), len(in_ranks))`.
    """
    op = operator.ge if eq else operator.gt
    return op(out_ranks[:, None], in_ranks)
