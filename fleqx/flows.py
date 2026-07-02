"""Constructors for normalizing-flow distributions.

Each constructor returns a plain `distreqx.distributions.Transformed` -- a base
distribution pushed through a bijector, no flow-specific wrapper class. As with any
distreqx distribution, `log_prob` and `sample` take a single event; use `jax.vmap`
for batches, e.g. `jax.vmap(flow.log_prob)(xs)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import parax
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
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from .bijectors import Coupling, Inverse, MaskedAutoregressive, Permute, Planar
from .bijectors._fork import require as _require_fork_bijector

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


class _SumLogDet(
    AbstractForwardInverseBijector, AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector
):
    """Wraps a bijector, summing its log-det to a scalar.

    `ScalarAffine`'s log-det is elementwise (one entry per dimension), unlike every
    other bijector `_finalize` composes it with via `Chain`, which already reduce to
    a scalar internally. Without summing here, `Chain` would silently broadcast-add
    a scalar and a `(dim,)` array, giving `log_prob` the wrong shape.

    `bijector` may itself be wrapped in a `parax.AbstractUnwrappable` (e.g.
    `parax.Freeze`, used by `_standardizing_bijector` below) -- resolved fresh via
    `parax.unwrap` on every call, not once in `__init__`, since `fit` reconstructs
    this bijector from scratch inside every differentiated step via `eqx.combine`; a
    `stop_gradient` from an earlier, unrelated trace has no effect on a later one.
    Doing this here rather than relying solely on `fit` to unwrap means a bijector
    like this also works standalone, with no separate unwrap step required.
    """

    bijector: AbstractBijector | parax.AbstractUnwrappable[AbstractBijector]
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijector: AbstractBijector | parax.AbstractUnwrappable[AbstractBijector]):
        self.bijector = bijector
        unwrapped = parax.unwrap(bijector)
        self._is_constant_jacobian = unwrapped.is_constant_jacobian
        self._is_constant_log_det = unwrapped.is_constant_log_det

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        y, log_det = parax.unwrap(self.bijector).forward_and_log_det(x)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x, log_det = parax.unwrap(self.bijector).inverse_and_log_det(y)
        return x, jnp.sum(log_det)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is _SumLogDet:
            return parax.unwrap(self.bijector).same_as(parax.unwrap(other.bijector))
        return False


def _standardizing_bijector(data: Float[Array, "n dim"]) -> AbstractBijector:
    loc = jnp.mean(data, axis=0)
    scale = jnp.std(data, axis=0)
    # Frozen: ScalarAffine stores scale, inv_scale and log_scale as independent
    # leaves that are supposed to satisfy scale == 1/inv_scale == exp(log_scale),
    # rather than deriving two of them from the third. MLE training only ever calls
    # `inverse_and_log_det` (`forward` is only used by `sample`), which touches
    # `inv_scale`/`log_scale` but never `scale` itself -- so leaving it trainable
    # would let gradient descent drift the two it does touch, desynchronizing all
    # three and breaking `forward(inverse(y)) == y`.
    return _SumLogDet(parax.Freeze(ScalarAffine(shift=loc, scale=scale)))


def _array_to_tree_bijector(template: PyTree[Array]) -> AbstractBijector:
    """A bijector mapping a flat 1D array to `template`'s PyTree structure.

    `template` is a single (unbatched) sample pytree of arrays; its leaves' shapes
    define the event shapes of the output. The resulting bijector's forward
    direction expects a 1D array whose length matches `template`'s total number of
    elements.

    Requires [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx):
    `Split`, `Reshape`, `Restructure` and `Leafwise` aren't in the PyPI release.
    """
    needed_by = "A pytree-shaped flow (`template=`)"
    Split = _require_fork_bijector("Split", needed_by=needed_by)
    Reshape = _require_fork_bijector("Reshape", needed_by=needed_by)
    Restructure = _require_fork_bijector("Restructure", needed_by=needed_by)
    Leafwise = _require_fork_bijector("Leafwise", needed_by=needed_by)

    leaves, treedef = jax.tree_util.tree_flatten(template)
    if not leaves:
        raise ValueError("`template` cannot be an empty PyTree.")

    event_shapes = [leaf.shape for leaf in leaves]
    leaf_sizes = [math.prod(shape) for shape in event_shapes]
    # Plain Python ints, not a jax/numpy array: `Split.indices_or_sections` is a
    # static (hashable) field.
    split_indices = np.cumsum(leaf_sizes)[:-1].tolist()

    split_bij = Split(indices_or_sections=split_indices, axis=-1)
    reshape_bijectors = tuple(
        Reshape(in_shape=(size,), out_shape=shape)
        for size, shape in zip(leaf_sizes, event_shapes)
    )
    leafwise_reshape_bij = Leafwise(reshape_bijectors)

    in_structure = tuple(range(len(event_shapes)))
    out_structure = jax.tree_util.tree_unflatten(treedef, in_structure)
    restructure_bij = Restructure(in_structure, out_structure)

    return Chain([restructure_bij, leafwise_reshape_bij, split_bij])


def _template_size(template: PyTree[Array]) -> int:
    leaves = jax.tree_util.tree_leaves(template)
    if not leaves:
        raise ValueError("`template` cannot be an empty PyTree.")
    return sum(math.prod(leaf.shape) for leaf in leaves)


def _flatten_pytree_data(data: PyTree[Float[Array, "n ..."]]) -> Float[Array, "n dim"]:
    """Ravels and concatenates each sample's leaves into a single flat vector."""

    def flatten_one(sample: PyTree[Array]) -> Array:
        return jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(sample)])

    return jax.vmap(flatten_one)(data)


def _resolve_dim(dim: int | None, template: PyTree[Array] | None) -> int:
    if (dim is None) == (template is None):
        raise ValueError("Exactly one of `dim` or `template` must be given.")
    return _template_size(template) if template is not None else dim


def _finalize(
    bijector: AbstractBijector,
    *,
    dim: int,
    data: PyTree[Float[Array, "n ..."]] | None,
    template: PyTree[Array] | None = None,
) -> Transformed:
    if data is not None:
        flat_data = _flatten_pytree_data(data) if template is not None else jnp.asarray(data)
        bijector = Chain([_standardizing_bijector(flat_data), bijector])
    if template is not None:
        bijector = Chain([_array_to_tree_bijector(template), bijector])
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
    dim: int | None = None,
    template: PyTree[Array] | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = relu,
    invert: bool = True,
    data: PyTree[Float[Array, "n ..."]] | None = None,
) -> Transformed:
    """Coupling flow ([Dinh et al., 2016](https://arxiv.org/abs/1605.08803)).

    A stack of affine coupling layers ([`fleqx.bijectors.Coupling`][]), each followed
    by a permutation, applied to a learnable diagonal-Gaussian base distribution.

    **Arguments:**

    - `key`: JAX random key.
    - `dim`: Dimensionality of the distribution. Exactly one of `dim` or `template`
        must be given.
    - `template`: A single (unbatched) sample pytree of arrays, given instead of
        `dim` for a distribution over an arbitrary pytree shape (e.g. a dict of
        named arrays) rather than a flat vector. `dim` is inferred as `template`'s
        total number of elements, and `data` (if given) should then be a pytree of
        the same structure, each leaf with a leading batch axis. Requires
        [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx). Defaults
        to `None`.
    - `flow_layers`: Number of coupling layers. Defaults to 8.
    - `nn_width`: Conditioner hidden layer width. Defaults to 50.
    - `nn_depth`: Conditioner depth. Defaults to 1.
    - `nn_activation`: Conditioner activation function. Defaults to `jax.nn.relu`.
    - `invert`: If `True` (default), `log_prob` is the fast direction; if `False`,
        `sample` is.
    - `data`: Optional array of shape `(n, dim)` (or a pytree matching `template`,
        if given). If given, an extra affine layer maps the flow's output to
        `data`'s mean and standard deviation from the start, rather than the base
        distribution learning this during training. This layer is frozen (`fit`
        won't move it further), so it's purely a fixed preprocessing step. Gives
        training a head start; useful when training for few epochs. Defaults to
        `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    dim = _resolve_dim(dim, template)
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
    return _finalize(bijector, dim=dim, data=data, template=template)


def masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    dim: int | None = None,
    template: PyTree[Array] | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = relu,
    invert: bool = True,
    data: PyTree[Float[Array, "n ..."]] | None = None,
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
    - `dim`: Dimensionality of the distribution. Exactly one of `dim` or `template`
        must be given.
    - `template`: A single (unbatched) sample pytree of arrays, given instead of
        `dim` for a distribution over an arbitrary pytree shape (e.g. a dict of
        named arrays) rather than a flat vector. `dim` is inferred as `template`'s
        total number of elements, and `data` (if given) should then be a pytree of
        the same structure, each leaf with a leading batch axis. Requires
        [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx). Defaults
        to `None`.
    - `flow_layers`: Number of masked autoregressive layers. Defaults to 8.
    - `nn_width`: Conditioner hidden layer width. Defaults to 50.
    - `nn_depth`: Conditioner depth. Defaults to 1.
    - `nn_activation`: Conditioner activation function. Defaults to `jax.nn.relu`.
    - `invert`: If `True` (default), `log_prob` is the fast (parallel) direction --
        recommended for maximum-likelihood training. If `False`, `sample` is fast
        instead.
    - `data`: Optional array of shape `(n, dim)` (or a pytree matching `template`,
        if given). If given, an extra affine layer maps the flow's output to
        `data`'s mean and standard deviation from the start, rather than the base
        distribution learning this during training. This layer is frozen (`fit`
        won't move it further), so it's purely a fixed preprocessing step. Gives
        training a head start; useful when training for few epochs. Defaults to
        `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    dim = _resolve_dim(dim, template)
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
    return _finalize(bijector, dim=dim, data=data, template=template)


def planar_flow(
    key: PRNGKeyArray,
    *,
    dim: int | None = None,
    template: PyTree[Array] | None = None,
    flow_layers: int = 8,
    negative_slope: float = 1e-2,
    invert: bool = True,
    data: PyTree[Float[Array, "n ..."]] | None = None,
) -> Transformed:
    """Planar flow ([Rezende and Mohamed, 2015](https://arxiv.org/abs/1505.05770)).

    A stack of planar layers ([`fleqx.bijectors.Planar`][]) applied to a learnable
    diagonal-Gaussian base distribution. No permutation is needed between layers,
    since each planar layer already depends on every dimension.

    **Arguments:**

    - `key`: JAX random key.
    - `dim`: Dimensionality of the distribution. Exactly one of `dim` or `template`
        must be given.
    - `template`: A single (unbatched) sample pytree of arrays, given instead of
        `dim` for a distribution over an arbitrary pytree shape (e.g. a dict of
        named arrays) rather than a flat vector. `dim` is inferred as `template`'s
        total number of elements, and `data` (if given) should then be a pytree of
        the same structure, each leaf with a leading batch axis. Requires
        [gvcallen's distreqx fork](https://github.com/gvcallen/distreqx). Defaults
        to `None`.
    - `flow_layers`: Number of planar layers. Defaults to 8.
    - `negative_slope`: Negative slope of the leaky ReLU used within each layer (see
        [`fleqx.bijectors.Planar`][]), in `(0, 1)`. Defaults to 0.01.
    - `invert`: If `True` (default), `log_prob` is the fast direction; if `False`,
        `sample` is.
    - `data`: Optional array of shape `(n, dim)` (or a pytree matching `template`,
        if given). If given, an extra affine layer maps the flow's output to
        `data`'s mean and standard deviation from the start, rather than the base
        distribution learning this during training. This layer is frozen (`fit`
        won't move it further), so it's purely a fixed preprocessing step. Gives
        training a head start; useful when training for few epochs. Defaults to
        `None`.

    **Returns:**

    A `distreqx.distributions.Transformed` distribution.
    """
    dim = _resolve_dim(dim, template)
    if dim < 1:
        raise ValueError(f"`dim` must be a positive integer, got {dim}.")

    def make_layer(k: PRNGKeyArray) -> AbstractBijector:
        return Planar(k, dim=dim, negative_slope=negative_slope)

    bijector = _stack_layers(key, flow_layers, make_layer)
    if invert:
        bijector = Inverse(bijector)
    return _finalize(bijector, dim=dim, data=data, template=template)

