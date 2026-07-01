"""Loss functions for training distributions."""

from __future__ import annotations

import equinox as eqx
import jax
from distreqx.distributions import AbstractDistribution
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


class MaximumLikelihoodLoss:
    """Negative log-likelihood loss for fitting a distribution by maximum likelihood.

    The call signature ``(params, static, x, key)`` matches what
    [`fleqx.train.fit`][] expects: ``params`` and ``static`` are the two halves of an
    ``equinox.partition`` of the distribution, ``x`` is a batch of observations, and
    ``key`` is accepted (and ignored) for API consistency with stochastic losses.
    """

    @eqx.filter_jit
    def __call__(
        self,
        params: PyTree,
        static: PyTree,
        x: Float[Array, "batch dim"],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Return the mean negative log-likelihood of ``x`` under the distribution."""
        dist: AbstractDistribution = eqx.combine(params, static)
        # `log_prob` takes a single, unbatched event -- like any distreqx
        # distribution -- so batches of `x` must be vmapped explicitly.
        return -jax.vmap(dist.log_prob)(x).mean()
