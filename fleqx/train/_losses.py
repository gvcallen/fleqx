"""Loss functions for training distributions."""

from __future__ import annotations

import equinox as eqx
import jax
from distreqx.distributions import AbstractDistribution
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


class MaximumLikelihoodLoss:
    """Negative log-likelihood loss for fitting a distribution by maximum likelihood.

    The call signature ``(model, x, key)`` matches what [`fleqx.train.fit`][]
    expects: ``model`` is the distribution being trained, ``x`` is a batch of
    observations (a pytree, matching whatever event shape ``model.log_prob``
    expects), and ``key`` is accepted (and ignored) for API consistency with
    stochastic losses.
    """

    @eqx.filter_jit
    def __call__(
        self,
        model: AbstractDistribution,
        x: PyTree[Float[Array, "batch ..."]],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Return the mean negative log-likelihood of ``x`` under ``model``."""
        # `log_prob` takes a single, unbatched event -- like any distreqx
        # distribution -- so batches of `x` must be vmapped explicitly.
        return -jax.vmap(model.log_prob)(x).mean()
