"""Fixed-permutation bijector."""

from __future__ import annotations

import jax.numpy as jnp
from distreqx.bijectors import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)
from jaxtyping import Array


class Permute(
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
):
    """Reorders the elements of a vector according to a fixed permutation.

    A permutation has a constant Jacobian determinant of magnitude 1, so it
    contributes nothing to a flow's log-density -- it exists purely to let later
    layers mix over dimensions left untouched by earlier ones.
    """

    permutation: Array
    inverse_permutation: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, permutation: Array):
        """Initializes a `Permute` bijector.

        **Arguments:**

        - `permutation`: A 1-D integer array containing a permutation of
            `arange(len(permutation))`.
        """
        self.permutation = jnp.asarray(permutation)
        self.inverse_permutation = jnp.argsort(self.permutation)
        self._is_constant_jacobian = True
        self._is_constant_log_det = True

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return x[self.permutation], jnp.zeros(())

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return y[self.inverse_permutation], jnp.zeros(())

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Permute:
            return self.permutation is other.permutation
        return False
