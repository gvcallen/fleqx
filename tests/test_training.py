"""Tests for the training loop and loss, using only the public fleqx API."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from distreqx.distributions import AbstractDistribution, Independent, Normal

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


def test_data_standardization_survives_training_unchanged():
    # Regression test: `ScalarAffine`'s `scale`, `inv_scale` and `log_scale` are
    # independent leaves that must satisfy scale == 1/inv_scale == exp(log_scale).
    # MLE training only ever exercises the bijector's inverse direction (`forward`
    # is only used by `sample`), which touches `inv_scale`/`log_scale` but never
    # `scale` -- so without freezing it, gradient descent would drift the two it
    # does touch and desynchronize all three, breaking `forward(inverse(y)) == y`.
    data = _target_data(jr.key(30), n=256) * jnp.array([1000.0, 0.01]) + jnp.array(
        [5000.0, -50.0]
    )
    flow = fleqx.flows.coupling_flow(
        jr.key(31), dim=DIM, flow_layers=2, nn_width=16, data=data
    )
    trained, _ = fit(
        jr.key(32), flow, data, max_epochs=10, batch_size=64, show_progress=False
    )

    initial_affine = flow.bijector.bijectors[0].bijector
    trained_affine = trained.bijector.bijectors[0].bijector
    for name in ("shift", "scale", "inv_scale", "log_scale"):
        assert jnp.array_equal(
            getattr(initial_affine, name), getattr(trained_affine, name)
        )

    y = jr.normal(jr.key(33), (DIM,)) * jnp.array([1000.0, 0.01]) + jnp.array(
        [5000.0, -50.0]
    )
    x = trained_affine.inverse(y)
    assert jnp.allclose(trained_affine.forward(x), y, atol=1e-3)


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


class _JointDistribution(eqx.Module):
    """Combines two distributions into one over `{"a": ..., "b": ...}` pytrees.

    Not part of the public `fleqx` API -- just enough structure to exercise `fit`
    with a pytree-shaped `dist`/`data`, standing in for e.g. a jointly-trained model
    over multiple named variables.
    """

    dist_a: AbstractDistribution
    dist_b: AbstractDistribution

    def sample(self, key):
        key_a, key_b = jr.split(key)
        return {"a": self.dist_a.sample(key_a), "b": self.dist_b.sample(key_b)}

    def log_prob(self, value):
        lp = self.dist_a.log_prob(value["a"])
        if value["b"] is not None:
            lp = lp + self.dist_b.log_prob(value["b"])
        return lp


class TestPytreeData:
    """`dist` and `data` can be arbitrary pytrees, not just a single array."""

    def _make_joint(self, key):
        key_a, key_b = jr.split(key)
        dist_a = fleqx.flows.coupling_flow(key_a, dim=DIM, flow_layers=2, nn_width=16)
        dist_b = Independent(Normal(loc=jnp.zeros(DIM), scale=jnp.ones(DIM)))
        return _JointDistribution(dist_a, dist_b)

    def test_fit_accepts_dict_pytree_data(self):
        joint = self._make_joint(jr.key(0))
        data = {
            "a": _target_data(jr.key(1), n=256),
            "b": _target_data(jr.key(2), n=256),
        }

        trained, losses = fit(
            jr.key(3),
            joint,
            data,
            learning_rate=1e-3,
            max_epochs=10,
            batch_size=64,
            show_progress=False,
        )

        assert all(jnp.isfinite(jnp.array(losses["train"])))
        assert losses["train"][-1] < losses["train"][0]

        sample = trained.sample(jr.key(4))
        assert set(sample) == {"a", "b"}
        assert jnp.isfinite(trained.log_prob(sample))

    def test_fit_accepts_none_leaf(self):
        # `dist_b` is untrained here -- `data["b"]` is `None` throughout, so only
        # `dist_a`'s parameters should move.
        joint = self._make_joint(jr.key(10))
        data = {"a": _target_data(jr.key(11), n=256), "b": None}

        trained, losses = fit(
            jr.key(12),
            joint,
            data,
            learning_rate=1e-3,
            max_epochs=10,
            batch_size=64,
            show_progress=False,
        )

        assert all(jnp.isfinite(jnp.array(losses["train"])))
        assert losses["train"][-1] < losses["train"][0]

        a_before = jax.tree_util.tree_leaves(eqx.filter(joint.dist_a, eqx.is_inexact_array))
        a_after = jax.tree_util.tree_leaves(eqx.filter(trained.dist_a, eqx.is_inexact_array))
        assert any(not jnp.array_equal(b, a) for b, a in zip(a_before, a_after))

        b_before = jax.tree_util.tree_leaves(eqx.filter(joint.dist_b, eqx.is_inexact_array))
        b_after = jax.tree_util.tree_leaves(eqx.filter(trained.dist_b, eqx.is_inexact_array))
        assert all(jnp.array_equal(b, a) for b, a in zip(b_before, b_after))

    def test_plain_array_data_is_unaffected(self):
        # The common case (a bare array, no surrounding pytree) is itself a trivial
        # pytree, so it should behave exactly as it did before this generalisation.
        flow = fleqx.flows.coupling_flow(jr.key(20), dim=DIM, flow_layers=2, nn_width=16)
        data = _target_data(jr.key(21), n=256)

        _, losses = fit(
            jr.key(22),
            flow,
            data,
            learning_rate=1e-3,
            max_epochs=10,
            batch_size=64,
            show_progress=False,
        )
        assert all(jnp.isfinite(jnp.array(losses["train"])))
        assert losses["train"][-1] < losses["train"][0]
