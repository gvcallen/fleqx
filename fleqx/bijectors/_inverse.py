"""Bijector inversion.

Vendored from [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx) as a
fallback for when it isn't installed -- see [`fleqx.bijectors`][] for how bijectors
are selected between the two.
"""

from __future__ import annotations

import equinox as eqx
from distreqx.bijectors import (
    AbstractBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)
from jaxtyping import PyTree


class Inverse(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector):
    """Inverts a bijector, swapping its forward and inverse directions.

    Bijectors are usually implemented with their cheaper direction as "forward". For
    a coupling flow, the inverse transform is used by `log_prob` and the forward
    transform by `sample`, so wrapping the flow's bijector in `Inverse` swaps which of
    the two is fast -- useful for prioritising fast maximum-likelihood training,
    which repeatedly evaluates `log_prob`.
    """

    bijector: AbstractBijector

    _is_constant_jacobian: bool = eqx.field(init=False)
    _is_constant_log_det: bool = eqx.field(init=False)

    def __post_init__(self):
        is_constant_jacobian = self.bijector.is_constant_jacobian
        is_constant_log_det = self.bijector.is_constant_log_det

        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )

        object.__setattr__(self, "_is_constant_jacobian", is_constant_jacobian)
        object.__setattr__(self, "_is_constant_log_det", is_constant_log_det)

    def forward(self, x: PyTree) -> PyTree:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: PyTree) -> PyTree:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:
            return self.bijector.same_as(other.bijector)
        return False
