"""Masked autoregressive bijector."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.bijectors import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)
from jaxtyping import Array, PRNGKeyArray

from ._affine_common import identity_init_offset, positive_scale
from ._masks import rank_based_mask


class MaskedAutoregressive(
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
):
    """Masked autoregressive bijector with an elementwise affine transform.

    See [Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057). An MLP with
    weights masked in a MADE-style (Germain et al., 2015) pattern maps `x` to a
    per-element shift and (positive) scale for each dimension, such that dimension
    `i`'s transform depends only on `x[:i]`. This makes evaluating the whole forward
    transform (and hence `log_prob`) a single parallel pass, at the cost of the
    inverse (hence `sample`) needing a `dim`-step sequential loop, one dimension at a
    time -- the opposite trade-off to [`fleqx.bijectors.Coupling`][].
    """

    layers: tuple[eqx.nn.Linear, ...]
    masks: tuple[Array, ...]
    activation: Callable
    dim: int
    min_scale: float
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jax.nn.relu,
        min_scale: float = 1e-2,
    ):
        """Initializes a `MaskedAutoregressive` bijector.

        **Arguments:**

        - `key`: JAX random key used to initialise the conditioner MLP.
        - `dim`: Total dimension.
        - `nn_width`: Conditioner hidden layer width.
        - `nn_depth`: Conditioner hidden layer depth.
        - `nn_activation`: Conditioner activation function. Defaults to
            `jax.nn.relu`.
        - `min_scale`: Lower bound added to the transform's scale, for numerical
            stability (also keeps the scale strictly positive). Defaults to 0.01.
        """
        mlp = eqx.nn.MLP(
            in_size=dim,
            out_size=2 * dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )
        in_ranks = jnp.arange(dim)
        # For dim==1 this would divide by zero; MAF has nothing to condition on
        # there regardless (all weights end up masked out below).
        hidden_ranks = jnp.arange(nn_width) % max(dim - 1, 1)
        out_ranks = jnp.repeat(jnp.arange(dim), 2)
        ranks = [in_ranks, *([hidden_ranks] * nn_depth), out_ranks]

        self.layers = mlp.layers
        self.masks = tuple(
            rank_based_mask(ranks[i], ranks[i + 1], eq=i != len(mlp.layers) - 1)
            for i in range(len(mlp.layers))
        )
        self.activation = nn_activation
        self.dim = dim
        self.min_scale = min_scale
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def _conditioner(self, x: Array) -> Array:
        h = x
        for i, (layer, mask) in enumerate(zip(self.layers, self.masks, strict=True)):
            h = jnp.where(mask, layer.weight, 0.0) @ h + layer.bias
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def _shift_and_scale(self, params: Array) -> tuple[Array, Array]:
        shift, raw_scale = jnp.reshape(params, (self.dim, 2)).T
        offset = identity_init_offset(self.min_scale)
        return shift, positive_scale(raw_scale + offset, self.min_scale)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        shift, scale = self._shift_and_scale(self._conditioner(x))
        y = x * scale + shift
        return y, jnp.sum(jnp.log(scale))

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|.

        Sequential in `dim`: recovering `x[i]` needs `x[:i]`, which is only
        available once earlier iterations have filled it in.
        """

        def step(x: Array, i: Array) -> tuple[Array, None]:
            shift, scale = self._shift_and_scale(self._conditioner(x))
            x = x.at[i].set((y[i] - shift[i]) / scale[i])
            return x, None

        x, _ = jax.lax.scan(step, jnp.zeros_like(y), jnp.arange(self.dim))
        _, scale = self._shift_and_scale(self._conditioner(x))
        return x, -jnp.sum(jnp.log(scale))

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return self is other
