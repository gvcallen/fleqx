"""Coupling bijector."""

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


class Coupling(
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
):
    """Coupling layer with an elementwise affine transform.

    See [Dinh et al., 2016](https://arxiv.org/abs/1605.08803). Splits the input into
    two parts. The first `untransformed_dim` elements are left untouched and fed
    through an MLP conditioner to produce a per-element shift and (positive) scale,
    which affinely transform the remaining elements.
    """

    untransformed_dim: int
    dim: int
    min_scale: float
    conditioner: eqx.nn.MLP
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        untransformed_dim: int,
        dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jax.nn.relu,
        min_scale: float = 1e-2,
    ):
        """Initializes a `Coupling` bijector.

        **Arguments:**

        - `key`: JAX random key used to initialise the conditioner MLP.
        - `untransformed_dim`: Number of leading elements left untouched.
        - `dim`: Total dimension.
        - `nn_width`: Conditioner hidden layer width.
        - `nn_depth`: Conditioner hidden layer depth.
        - `nn_activation`: Conditioner activation function. Defaults to
            `jax.nn.relu`.
        - `min_scale`: Lower bound added to the transform's scale, for numerical
            stability (also keeps the scale strictly positive). Defaults to 0.01.
        """
        self.untransformed_dim = untransformed_dim
        self.dim = dim
        self.min_scale = min_scale
        transform_dim = dim - untransformed_dim
        self.conditioner = eqx.nn.MLP(
            in_size=untransformed_dim,
            out_size=2 * transform_dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def _shift_and_scale(self, x_cond: Array) -> tuple[Array, Array]:
        # Each dimension's (shift, raw_scale) pair is adjacent in the conditioner's
        # output, not grouped by kind -- i.e. [shift_0, scale_0, shift_1, scale_1, ...].
        shift, raw_scale = jnp.reshape(self.conditioner(x_cond), (-1, 2)).T
        offset = identity_init_offset(self.min_scale)
        return shift, positive_scale(raw_scale + offset, self.min_scale)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        x_cond, x_trans = x[: self.untransformed_dim], x[self.untransformed_dim :]
        shift, scale = self._shift_and_scale(x_cond)
        y = jnp.concatenate([x_cond, x_trans * scale + shift])
        return y, jnp.sum(jnp.log(scale))

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        y_cond, y_trans = y[: self.untransformed_dim], y[self.untransformed_dim :]
        shift, scale = self._shift_and_scale(y_cond)
        x = jnp.concatenate([y_cond, (y_trans - shift) / scale])
        return x, -jnp.sum(jnp.log(scale))

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return self is other
