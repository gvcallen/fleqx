"""Generic training utilities: batching, splitting, and an optax step.

Nothing here is specific to distreqx or to flows -- these operate on any pytree and
loss function. `data` pytrees may contain `None` leaves (e.g. an unused optional
field); `jax.tree_util` treats those as empty subtrees, so they pass through
untouched wherever a real array would be shuffled, split or batched.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import parax
from jax import jit
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar


def combine_and_unwrap(params: PyTree, static: PyTree) -> PyTree:
    """Reconstructs a model from `params`/`static`, resolving any parax wrappers.

    `parax.unwrap` has to be called fresh on every reconstruction (not once when the
    model was first built), since `fit` calls this inside every differentiated step
    -- a `stop_gradient` from an earlier, unrelated trace has no effect on a later
    one.
    """
    return parax.unwrap(eqx.combine(params, static))


@eqx.filter_jit
def step(
    params: PyTree,
    static: PyTree,
    *args,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable[..., Scalar],
    **kwargs,
) -> tuple[PyTree, PyTree, Scalar]:
    """A single optimisation step.

    Calls `loss_fn(model, *args, **kwargs)`, where `model` is reconstructed from
    `params`/`static` via `combine_and_unwrap`.
    """

    def _loss(params: PyTree, *args, **kwargs) -> Scalar:
        return loss_fn(combine_and_unwrap(params, static), *args, **kwargs)

    loss_val, grads = eqx.filter_value_and_grad(_loss)(params, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


@eqx.filter_jit
def evaluate(
    params: PyTree,
    static: PyTree,
    *args,
    loss_fn: Callable[..., Scalar],
    **kwargs,
) -> Scalar:
    """Evaluates `loss_fn` without computing gradients or updating `params`.

    Otherwise identical to `step`; used for validation.
    """
    return loss_fn(combine_and_unwrap(params, static), *args, **kwargs)


def leading_axis_size(data: PyTree) -> int:
    """The shared leading-axis size of `data`'s (non-`None`) leaves."""
    leaves = jax.tree_util.tree_leaves(data)
    if not leaves:
        raise ValueError("`data` has no array leaves to determine a sample count from.")
    return leaves[0].shape[0]


def shuffle(key: PRNGKeyArray, data: PyTree) -> PyTree:
    """Shuffles every leaf of `data` along axis 0, with a shared permutation."""
    perm = jr.permutation(key, leading_axis_size(data))
    return jax.tree_util.tree_map(lambda a: a[perm], data)


def train_val_split(
    key: PRNGKeyArray,
    data: PyTree,
    val_prop: float = 0.1,
) -> tuple[PyTree, PyTree]:
    """Random train/validation split of a pytree of arrays sharing axis-0 size."""
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    n_train = leading_axis_size(data) - round(val_prop * leading_axis_size(data))
    data = shuffle(key, data)
    train_data = jax.tree_util.tree_map(lambda a: a[:n_train], data)
    val_data = jax.tree_util.tree_map(lambda a: a[n_train:], data)
    return train_data, val_data


@partial(jit, static_argnums=1)
def get_batches(data: PyTree, batch_size: int) -> PyTree:
    """Reshape `data`'s leaves with shape ``(n, ...)`` to ``(n // batch_size,
    batch_size, ...)``.

    The trailing partial batch is dropped if truncated (to avoid recompilation), and
    `batch_size` is capped at the leading axis size.
    """
    batch_size = min(batch_size, leading_axis_size(data))
    return jax.tree_util.tree_map(lambda a: _add_batch(a, batch_size), data)


def _add_batch(arr: Array, batch_size: int) -> Array:
    n_batches = arr.shape[0] // batch_size
    return arr[: n_batches * batch_size].reshape(n_batches, batch_size, *arr.shape[1:])


def count_fruitless(losses: list[float]) -> int:
    """Number of epochs since the minimum loss in a list of losses."""
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
