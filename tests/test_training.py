"""Tests for the training loop and loss, using only the public fleqx API."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import fleqx
from fleqx.train import MaximumLikelihoodLoss, fit

DIM = 2

FLOW_CONSTRUCTORS = {
    "coupling": (fleqx.flows.coupling_flow, {"nn_width": 32}),
    "masked_autoregressive": (fleqx.flows.masked_autoregressive_flow, {"nn_width": 32}),
    "planar": (fleqx.flows.planar_flow, {}),
}


def _target_data(key, n=512):
    """Samples from a simple anisotropic, shifted Gaussian."""
    loc = jnp.array([2.0, -1.0])
    scale = jnp.array([0.5, 1.5])
    return loc + scale * jr.normal(key, (n, DIM))


def test_training_reduces_loss():
    key = jr.key(0)
    data = _target_data(jr.key(1))
    flow = fleqx.flows.coupling_flow(key, dim=DIM, flow_layers=4, nn_width=32)

    trained, losses = fit(
        jr.key(2),
        flow,
        data,
        learning_rate=1e-3,
        max_epochs=15,
        batch_size=128,
        show_progress=False,
    )

    assert set(losses) == {"train", "val"}
    assert len(losses["train"]) == len(losses["val"])
    assert len(losses["train"]) <= 15
    assert all(jnp.isfinite(jnp.array(losses["train"])))
    assert all(jnp.isfinite(jnp.array(losses["val"])))
    # The flow should learn something: final loss well below the initial loss.
    assert losses["train"][-1] < losses["train"][0]
    assert isinstance(trained, type(flow))


@pytest.mark.parametrize("flow_name", FLOW_CONSTRUCTORS)
def test_training_reduces_loss_for_every_flow_type(flow_name):
    ctor, kwargs = FLOW_CONSTRUCTORS[flow_name]
    data = _target_data(jr.key(20))
    flow = ctor(jr.key(21), dim=DIM, flow_layers=4, **kwargs)

    _, losses = fit(
        jr.key(22),
        flow,
        data,
        learning_rate=1e-3,
        max_epochs=15,
        batch_size=128,
        show_progress=False,
    )
    assert all(jnp.isfinite(jnp.array(losses["train"])))
    assert losses["train"][-1] < losses["train"][0]


def test_trained_flow_is_usable():
    data = _target_data(jr.key(3))
    flow = fleqx.flows.coupling_flow(jr.key(4), dim=DIM, flow_layers=2, nn_width=16)
    trained, _ = fit(
        jr.key(5), flow, data, max_epochs=3, batch_size=128, show_progress=False
    )

    sample = trained.sample(jr.key(6))
    assert sample.shape == (DIM,)
    assert jnp.isfinite(trained.log_prob(sample))


def test_loss_is_scalar_and_finite():
    flow = fleqx.flows.coupling_flow(jr.key(7), dim=DIM, flow_layers=2, nn_width=16)
    data = _target_data(jr.key(8), n=64)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    loss = MaximumLikelihoodLoss()(params, static, data)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


class _NoisyLoss:
    """Wraps a loss with large, key-driven noise, injecting non-monotonicity into a
    validation trajectory without needing a destabilizingly high learning rate (the
    flow's base distribution has an unconstrained -- not positivity-reparameterized --
    scale, so an aggressive learning rate can drive it to a degenerate value)."""

    def __init__(self, base_loss, noise_scale):
        self.base_loss = base_loss
        self.noise_scale = noise_scale

    def __call__(self, params, static, x, key):
        noise = self.noise_scale * jax.random.normal(key)
        return self.base_loss(params, static, x, key) + noise


def test_early_stopping_respects_max_patience():
    data = _target_data(jr.key(9), n=256)
    flow = fleqx.flows.coupling_flow(jr.key(10), dim=DIM, flow_layers=2, nn_width=16)

    _, losses = fit(
        jr.key(11),
        flow,
        data,
        loss_fn=_NoisyLoss(MaximumLikelihoodLoss(), noise_scale=10.0),
        max_epochs=200,
        max_patience=2,
        batch_size=64,
        show_progress=False,
    )
    assert len(losses["val"]) < 200


def test_return_best_differs_from_final_when_trajectory_is_noisy():
    data = _target_data(jr.key(12), n=256)
    flow = fleqx.flows.coupling_flow(jr.key(13), dim=DIM, flow_layers=2, nn_width=16)
    fit_kwargs = dict(
        loss_fn=_NoisyLoss(MaximumLikelihoodLoss(), noise_scale=10.0),
        max_epochs=30,
        max_patience=30,  # effectively disable early stopping
        batch_size=64,
        show_progress=False,
    )

    best, losses = fit(jr.key(14), flow, data, return_best=True, **fit_kwargs)
    final, _ = fit(jr.key(14), flow, data, return_best=False, **fit_kwargs)

    # A noisy trajectory (the best epoch's validation loss strictly beats the final
    # epoch's) means the best and final parameters are genuinely different -- proving
    # `return_best` has a real effect, rather than trivially matching `final`.
    assert min(losses["val"]) < losses["val"][-1]
    best_leaves = jax.tree_util.tree_leaves(eqx.filter(best, eqx.is_inexact_array))
    final_leaves = jax.tree_util.tree_leaves(eqx.filter(final, eqx.is_inexact_array))
    assert any(
        not jnp.array_equal(b, f) for b, f in zip(best_leaves, final_leaves)
    )
