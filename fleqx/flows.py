"""Constructors for normalizing-flow distributions.

Each constructor returns a plain `distreqx.distributions.Transformed` -- a base
distribution pushed through a bijector, no flow-specific wrapper class. As with any
distreqx distribution, `log_prob` and `sample` take a single event; use `jax.vmap`
for batches, e.g. `jax.vmap(flow.log_prob)(xs)`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distreqx.bijectors import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
    Chain,
    ScalarAffine,
)
from distreqx.distributions import Independent, Normal, Transformed
from jax.nn import relu
from jaxtyping import Array, Float, PRNGKeyArray

from .bijectors import Coupling, Inverse, MaskedAutoregressive, Permute, Planar

__all__ = ["coupling_flow", "masked_autoregressive_flow", "planar_flow"]


def _default_permute(dim: int, key: PRNGKeyArray) -> AbstractBijector | None:
    """A permutation to mix dimensions between layers.

    `None` for `dim == 1`. For `dim == 2`, always swaps rather than drawing a random
    permutation, since a random permutation of two elements is the identity half the
    time.
    """
    if dim == 1:
        return None
    if dim == 2:
        return Permute(jnp.array([1, 0]))
    return Permute(jr.permutation(key, dim))


class _FrozenSumLogDet(
    AbstractForwardInverseBijector, AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector
):
    """Wraps a bijector, freezing its parameters and summing its log-det to a scalar.

    Two independent fixes bundled into one wrapper, both needed for `ScalarAffine`
    specifically:

    - Its log-det is elementwise (one entry per dimension), unlike every other
      bijector `_finalize` composes it with via `Chain`, which already reduce to a
      scalar internally. Without summing here, `Chain` would silently broadcast-add
      a scalar and a `(dim,)` array, giving `log_prob` the wrong shape.
    - It stores `scale`, `inv_scale` and `log_scale` as independent leaves that are
      supposed to satisfy `scale == 1/inv_scale == exp(log_scale)`, rather than
      deriving two of them from the third. MLE training only ever calls
      `inverse_and_log_det` (`forward` is only used by `sample`), which touches
      `inv_scale`/`log_scale` but never `scale` itself -- so gradient descent would
      drift the two it does touch, silently desynchronizing all three and breaking
      `forward(inverse(y)) == y`. Stopping the gradient here keeps it a fixed,
      self-consistent preprocessing step, initialised once from `data`.

    `stop_gradient` has to be applied on every call (not once in `__init__`) since
    `fit` reconstructs this bijector from scratch inside every differentiated step
    via `eqx.combine` -- a `stop_gradient` from an earlier, unrelated trace has no
    effect on a later one.
    """

    bijector: AbstractBijector
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijector: AbstractBijector):
        self.bijector = bijector
        self._is_constant_jacobian = bijector.is_constant_jacobian
        self._is_constant_log_det = bijector.is_constant_log_det

    def _frozen_bijector(self) -> AbstractBijector:
        params, static = eqx.partition(self.bijector, eqx.is_inexact_array)
        return eqx.combine(jax.tree_util.tree_map(jax.lax.stop_gradient, params), static)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        y, log_det = self._frozen_bijector().forward_and_log_det(x)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x, log_det = self._frozen_bijector().inverse_and_log_det(y)
        return x, jnp.sum(log_det)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is _FrozenSumLogDet:
            return self.bijector.same_as(other.bijector)
        return False


def _standardizing_bijector(data: Float[Array, "n dim"]) -> AbstractBijector:
    loc = jnp.mean(data, axis=0)
    scale = jnp.std(data, axis=0)
    return _FrozenSumLogDet(ScalarAffine(shift=loc, scale=scale))


def _finalize(
    bijector: AbstractBijector, *, dim: int, data: Float[Array, "n dim"] | None
) -> Transformed:
    if data is not None:
        bijector = Chain([_standardizing_bijector(jnp.asarray(data)), bijector])
    base = Independent(Normal(loc=jnp.zeros(dim), scale=jnp.ones(dim)))
    return Transformed(base, bijector)


def _stack_layers(
    key: PRNGKeyArray, flow_layers: int, make_layer: Callable[[PRNGKeyArray], AbstractBijector]
) -> AbstractBijector:
    layer_keys = jr.split(key, flow_layers)
    layers = [make_layer(k) for k in layer_keys]
    return Chain(list(reversed(layers)))


def coupling_flow(
    key: PRNGKeyArray,
    *,
    dim: int,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = relu,
    invert: bool = True,
    data: Float[Array, "n dim"] | None = None,
) -> Transformed:
    """Coupling flow ([Dinh et al., 2016](https://arxiv.org/abs/1605.08803)).

    A stack of affine coupling layers ([`fleqx.bijectors.Coupling`][]), each followed
    by a permutation, applied to a learnable diagonal-Gaussian base distribution.

    **Arguments:**

    - `key`: JAX random key.
    - `dim`: Dimensionality of the distribution.
    - `flow_layers`: Number of coupling layers. Defaults to 8.
    - `nn_width`: Conditioner hidden layer width. Defaults to 50.
    - `nn_depth`: Conditioner depth. Defaults to 1.
    - `nn_activation`: Conditioner activation function. Defaults to `jax.nn.relu`.
    - `invert`: If `True` (default), `log_prob` is the fast direction; if `False`,
        `sample` is.
    - `data`: Optional array of shape `(n, dim)`. If given, an extra affine
        layer maps the flow's output to `data`'s mean and standard deviation
        from the start, rather than the base distribution learning this during
        training. This layer is frozen (`fit` won't move it further), so it's
        purely a fixed preprocessing step. Gives training a head start; useful
        when training for few epochs. Defaults to `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    if dim < 1:
        raise ValueError(f"`dim` must be a positive integer, got {dim}.")

    def make_layer(k: PRNGKeyArray) -> AbstractBijector:
        bij_key, perm_key = jr.split(k)
        coupling = Coupling(
            bij_key,
            untransformed_dim=dim // 2,
            dim=dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        permute = _default_permute(dim, perm_key)
        return coupling if permute is None else Chain([permute, coupling])

    bijector = _stack_layers(key, flow_layers, make_layer)
    if invert:
        bijector = Inverse(bijector)
    return _finalize(bijector, dim=dim, data=data)


def masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    dim: int,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = relu,
    invert: bool = True,
    data: Float[Array, "n dim"] | None = None,
) -> Transformed:
    """Masked autoregressive flow ([Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057)).

    A stack of masked autoregressive layers ([`fleqx.bijectors.MaskedAutoregressive`][]),
    each followed by a permutation, applied to a learnable diagonal-Gaussian base
    distribution.

    Unlike [`coupling_flow`][fleqx.flows.coupling_flow], the two transform directions
    genuinely differ in cost: one is a single parallel pass, the other a `dim`-step
    sequential loop. `invert` picks which direction `log_prob` gets.

    **Arguments:**

    - `key`: JAX random key.
    - `dim`: Dimensionality of the distribution.
    - `flow_layers`: Number of masked autoregressive layers. Defaults to 8.
    - `nn_width`: Conditioner hidden layer width. Defaults to 50.
    - `nn_depth`: Conditioner depth. Defaults to 1.
    - `nn_activation`: Conditioner activation function. Defaults to `jax.nn.relu`.
    - `invert`: If `True` (default), `log_prob` is the fast (parallel) direction --
        recommended for maximum-likelihood training. If `False`, `sample` is fast
        instead.
    - `data`: Optional array of shape `(n, dim)`. If given, an extra affine
        layer maps the flow's output to `data`'s mean and standard deviation
        from the start, rather than the base distribution learning this during
        training. This layer is frozen (`fit` won't move it further), so it's
        purely a fixed preprocessing step. Gives training a head start; useful
        when training for few epochs. Defaults to `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    if dim < 1:
        raise ValueError(f"`dim` must be a positive integer, got {dim}.")

    def make_layer(k: PRNGKeyArray) -> AbstractBijector:
        bij_key, perm_key = jr.split(k)
        maf = MaskedAutoregressive(
            bij_key,
            dim=dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        permute = _default_permute(dim, perm_key)
        return maf if permute is None else Chain([permute, maf])

    bijector = _stack_layers(key, flow_layers, make_layer)
    if invert:
        bijector = Inverse(bijector)
    return _finalize(bijector, dim=dim, data=data)


def planar_flow(
    key: PRNGKeyArray,
    *,
    dim: int,
    flow_layers: int = 8,
    negative_slope: float = 1e-2,
    invert: bool = True,
    data: Float[Array, "n dim"] | None = None,
) -> Transformed:
    """Planar flow ([Rezende and Mohamed, 2015](https://arxiv.org/abs/1505.05770)).

    A stack of planar layers ([`fleqx.bijectors.Planar`][]) applied to a learnable
    diagonal-Gaussian base distribution. No permutation is needed between layers,
    since each planar layer already depends on every dimension.

    **Arguments:**

    - `key`: JAX random key.
    - `dim`: Dimensionality of the distribution.
    - `flow_layers`: Number of planar layers. Defaults to 8.
    - `negative_slope`: Negative slope of the leaky ReLU used within each layer (see
        [`fleqx.bijectors.Planar`][]), in `(0, 1)`. Defaults to 0.01.
    - `invert`: If `True` (default), `log_prob` is the fast direction; if `False`,
        `sample` is.
    - `data`: Optional array of shape `(n, dim)`. If given, an extra affine
        layer maps the flow's output to `data`'s mean and standard deviation
        from the start, rather than the base distribution learning this during
        training. This layer is frozen (`fit` won't move it further), so it's
        purely a fixed preprocessing step. Gives training a head start; useful
        when training for few epochs. Defaults to `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    if dim < 1:
        raise ValueError(f"`dim` must be a positive integer, got {dim}.")

    def make_layer(k: PRNGKeyArray) -> AbstractBijector:
        return Planar(k, dim=dim, negative_slope=negative_slope)

    bijector = _stack_layers(key, flow_layers, make_layer)
    if invert:
        bijector = Inverse(bijector)
    return _finalize(bijector, dim=dim, data=data)

