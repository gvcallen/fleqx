"""Bijector inversion."""

from __future__ import annotations

from distreqx.bijectors import AbstractBijector
from jaxtyping import Array


class Invert(AbstractBijector):
    """Inverts a bijector, swapping its forward and inverse directions.

    Bijectors are usually implemented with their cheaper direction as "forward". For
    a coupling flow, the inverse transform is used by `log_prob` and the forward
    transform by `sample`, so wrapping the flow's bijector in `Invert` swaps which of
    the two is fast -- useful for prioritising fast maximum-likelihood training,
    which repeatedly evaluates `log_prob`.
    """

    bijector: AbstractBijector
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijector: AbstractBijector):
        """Initializes an `Invert` bijector.

        **Arguments:**

        - `bijector`: The bijector to invert.
        """
        self.bijector = bijector
        self._is_constant_jacobian = bijector.is_constant_jacobian
        self._is_constant_log_det = bijector.is_constant_log_det

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return self.bijector.inverse_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return self.bijector.forward_log_det_jacobian(y)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Invert:
            return self.bijector.same_as(other.bijector)
        return False
