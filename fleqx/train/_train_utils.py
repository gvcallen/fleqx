"""Generic training utilities: batching, splitting, and an optax step.

Nothing here is specific to distreqx or to flows -- these operate on any pytree and
loss function.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar


@eqx.filter_jit
def step(
    params: PyTree,
    *args,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable[..., Scalar],
    **kwargs,
) -> tuple[PyTree, PyTree, Scalar]:
    """A single optimisation step."""
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


def train_val_split(
    key: PRNGKeyArray,
    arrays: Sequence[Array],
    val_prop: float = 0.1,
) -> tuple[list[Array], list[Array]]:
    """Random train/validation split for a sequence of arrays sharing axis-0 size."""
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    num_samples = arrays[0].shape[0]
    n_train = num_samples - round(val_prop * num_samples)
    arrays = [jr.permutation(key, a) for a in arrays]
    train_arrays = [arr[:n_train] for arr in arrays]
    val_arrays = [arr[n_train:] for arr in arrays]
    return train_arrays, val_arrays


@partial(jit, static_argnums=1)
def get_batches(arrays: Sequence[Array], batch_size: int) -> tuple[Array, ...]:
    """Reshape arrays with shape ``(n, ...)`` to ``(n // batch_size, batch_size, ...)``.

    The trailing partial batch is dropped if truncated (to avoid recompilation), and
    `batch_size` is capped at the array length.
    """
    return tuple(_add_batch(arr, batch_size) for arr in arrays)


def _add_batch(arr: Array, batch_size: int) -> Array:
    batch_size = min(batch_size, arr.shape[0])
    n_batches = arr.shape[0] // batch_size
    return arr[: n_batches * batch_size].reshape(n_batches, batch_size, *arr.shape[1:])


def count_fruitless(losses: list[float]) -> int:
    """Number of epochs since the minimum loss in a list of losses."""
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
