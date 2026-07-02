"""Fixed-permutation bijector.

Vendored from [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx) as a
fallback for when it isn't installed -- see [`fleqx.bijectors`][] for how bijectors
are selected between the two.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from distreqx.bijectors import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)
from jaxtyping import Array


class Permute(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
):
    """Reorders the elements of a vector according to a fixed permutation.

    A permutation has a constant Jacobian determinant of magnitude 1, so it
    contributes nothing to a flow's log-density -- it exists purely to let later
    layers mix over dimensions left untouched by earlier ones.
    """

    permutation: Array
    inverse_permutation: Array

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def __init__(self, permutation: Array | list | tuple):
        """Initializes a `Permute` bijector.

        **Arguments:**

        - `permutation`: An array or list of integers representing the new order
            of the elements. Must contain all integers from 0 to N-1 exactly once.
        """
        # Convert to numpy first to validate shapes/values before JIT compilation.
        perm_np = np.asarray(permutation, dtype=int)

        if perm_np.ndim != 1:
            raise ValueError(
                f"Permutation must be a 1D array, got shape {perm_np.shape}."
            )

        expected_elements = np.arange(perm_np.size)
        if not np.array_equal(np.sort(perm_np), expected_elements):
            raise ValueError(
                "Invalid permutation. It must contain all integers from "
                f"0 to {perm_np.size - 1} exactly once."
            )

        self.permutation = jnp.asarray(perm_np)
        self.inverse_permutation = jnp.argsort(self.permutation)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = x[permutation] and log|det J(f)(x)| = 0.0."""
        return x[self.permutation], jnp.zeros((), dtype=x.dtype)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = y[inverse_permutation] and log|det J(f^{-1})(y)| = 0.0."""
        return y[self.inverse_permutation], jnp.zeros((), dtype=y.dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return bool(
            type(other) is Permute
            and jnp.array_equal(self.permutation, other.permutation)
        )
