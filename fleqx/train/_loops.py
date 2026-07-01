"""Training loops."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from distreqx.distributions import AbstractDistribution
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from tqdm import tqdm

from ._losses import MaximumLikelihoodLoss
from ._train_utils import count_fruitless, get_batches, step, train_val_split


def fit(
    key: PRNGKeyArray,
    dist: AbstractDistribution,
    data: Float[Array, "n dim"],
    *,
    loss_fn: Callable[..., Scalar] | None = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[AbstractDistribution, dict[str, list[float]]]:
    """Fit a distribution to samples by minimising a loss (maximum likelihood by default).

    A held-out validation split is used for early stopping: training stops once
    `max_patience` consecutive epochs pass without an improvement in validation loss.
    The last batch in each epoch is dropped if it would be truncated, to avoid
    recompilation.

    **Arguments:**

    - `key`: JAX random key controlling the train/validation split, shuffling, and
        (if applicable) any stochasticity in `loss_fn`.
    - `dist`: The distribution to train (as returned by e.g.
        [`fleqx.flows.coupling_flow`][]).
    - `data`: Array of observations with shape ``(n, dim)``.
    - `loss_fn`: Loss with signature ``(params, static, x, key)``. Defaults to
        [`fleqx.train.MaximumLikelihoodLoss`][].
    - `learning_rate`: Adam learning rate. Ignored if `optimizer` is given.
    - `optimizer`: An optax optimizer. Defaults to ``optax.adam(learning_rate)``.
    - `max_epochs`: Maximum number of passes over the data. Defaults to 100.
    - `max_patience`: Number of consecutive epochs with no validation-loss
        improvement after which training stops early. Defaults to 5.
    - `batch_size`: Mini-batch size. Defaults to 100.
    - `val_prop`: Proportion of `data` held out for validation. Defaults to 0.1.
    - `return_best`: Whether to return the parameters from the epoch with the lowest
        validation loss (`True`), or the parameters after the final update (`False`).
        Defaults to `True`.
    - `show_progress`: Whether to display a progress bar. Defaults to `True`.

    **Returns:**

    - A tuple ``(trained_dist, losses)``, where ``losses`` is a dict with ``"train"``
        and ``"val"`` keys, each holding the mean loss per epoch.
    """
    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    data = (jnp.asarray(data),)
    params, static = eqx.partition(dist, eqx.is_inexact_array)
    best_params = params
    opt_state = optimizer.init(params)

    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses: dict[str, list[float]] = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)
    for _ in loop:
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            params, opt_state, loss_i = step(
                params,
                static,
                *batch,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=subkey,
            )
            batch_losses.append(loss_i)
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = eqx.filter_jit(loss_fn)(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params
        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    trained = eqx.combine(params, static)
    return trained, losses
