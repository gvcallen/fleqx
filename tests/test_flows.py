"""Tests for flow construction and the distreqx distribution interface.

These tests only use the public ``fleqx`` API together with ``distreqx`` and ``jax``.
They are intentionally agnostic to how each flow is implemented under the hood: a
``fleqx`` flow is a plain ``distreqx.distributions.Transformed``, and nothing here
should require knowledge of what produced its bijector. Every constructor in
``fleqx.flows`` is exercised via the same parametrized tests, plus a few
flow-specific extras.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from distreqx import bijectors as distreqx_bijectors
from distreqx.bijectors import AbstractBijector
from distreqx.distributions import AbstractDistribution, Transformed

import fleqx

DIM = 3

_FORK_INSTALLED = hasattr(distreqx_bijectors, "Split")

# (constructor, extra kwargs beyond `key`, `dim`, `flow_layers`, `data`) for each flow
# type. Not every constructor takes the same conditioner-network arguments (e.g.
# `planar_flow` has no MLP conditioner at all), so kwargs are looked up per flow.
FLOW_CONSTRUCTORS = {
    "coupling": (fleqx.flows.coupling_flow, {"nn_width": 16}),
    "masked_autoregressive": (fleqx.flows.masked_autoregressive_flow, {"nn_width": 16}),
    "planar": (fleqx.flows.planar_flow, {}),
}


def _make_flow(name, key, *, dim=DIM, flow_layers=2, **overrides):
    ctor, kwargs = FLOW_CONSTRUCTORS[name]
    return ctor(key, dim=dim, flow_layers=flow_layers, **{**kwargs, **overrides})


@pytest.fixture(params=list(FLOW_CONSTRUCTORS), ids=list(FLOW_CONSTRUCTORS))
def flow_name(request):
    return request.param


@pytest.fixture
def flow(flow_name):
    return _make_flow(flow_name, jr.key(0))


def test_is_distreqx_distribution(flow):
    assert isinstance(flow, AbstractDistribution)
    assert isinstance(flow, Transformed)


def test_is_composed_of_distreqx_primitives(flow):
    # A fleqx flow is a base distribution + bijector, like any other distreqx
    # `Transformed`, not a bespoke flow class.
    assert isinstance(flow.distribution, AbstractDistribution)
    assert isinstance(flow.bijector, AbstractBijector)


def test_event_shape(flow):
    assert flow.event_shape == (DIM,)


def test_sample_shape(flow):
    sample = flow.sample(jr.key(1))
    assert sample.shape == (DIM,)
    assert jnp.all(jnp.isfinite(sample))


def test_log_prob_scalar(flow):
    x = flow.sample(jr.key(2))
    lp = flow.log_prob(x)
    assert lp.shape == ()
    assert jnp.isfinite(lp)


def test_log_prob_batches(flow):
    # As with any distreqx distribution, `log_prob` takes a single event; batches are
    # handled with `jax.vmap`, not by passing in an array with a leading batch axis.
    xs = jax.vmap(flow.sample)(jr.split(jr.key(3), 8))
    assert xs.shape == (8, DIM)
    lps = jax.vmap(flow.log_prob)(xs)
    assert lps.shape == (8,)
    assert jnp.all(jnp.isfinite(lps))


def test_sample_and_log_prob_consistent(flow):
    sample, lp = flow.sample_and_log_prob(jr.key(4))
    assert sample.shape == (DIM,)
    assert jnp.allclose(lp, flow.log_prob(sample), atol=1e-4)


def test_prob_matches_exp_log_prob(flow):
    x = flow.sample(jr.key(5))
    assert jnp.allclose(flow.prob(x), jnp.exp(flow.log_prob(x)), atol=1e-5)


def test_is_pytree_with_trainable_params(flow):
    leaves = eqx.filter(flow, eqx.is_inexact_array, is_leaf=lambda x: x is None)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(leaves) if leaf is not None]
    assert len(leaves) > 0
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves)


def test_different_keys_give_different_flows(flow_name):
    x = jnp.zeros(DIM)
    f0 = _make_flow(flow_name, jr.key(0))
    f1 = _make_flow(flow_name, jr.key(1))
    assert not jnp.allclose(f0.log_prob(x), f1.log_prob(x))


def test_invalid_dim_raises(flow_name):
    ctor, kwargs = FLOW_CONSTRUCTORS[flow_name]
    with pytest.raises(ValueError):
        ctor(jr.key(0), dim=0, **kwargs)


def test_dim_and_template_are_mutually_exclusive(flow_name):
    ctor, kwargs = FLOW_CONSTRUCTORS[flow_name]
    with pytest.raises(ValueError, match="Exactly one"):
        ctor(jr.key(0), **kwargs)  # neither given
    with pytest.raises(ValueError, match="Exactly one"):
        ctor(jr.key(0), dim=DIM, template=jnp.zeros(DIM), **kwargs)  # both given


@pytest.mark.parametrize("method", ["mean", "mode", "entropy"])
def test_undefined_moments_raise(flow, method):
    # None of these flows have a constant-Jacobian bijector, so these are genuinely
    # undefined, matching the behaviour of `distreqx.distributions.Transformed`
    # generally.
    with pytest.raises(NotImplementedError):
        getattr(flow, method)()


def test_data_standardization_moves_scale_toward_target(flow_name):
    # A tight absolute check on an untrained flow's sample scale is not robust: with
    # the default `invert=True`, `sample` uses each layer's *inverse* direction,
    # which can divide by an untrained, unconstrained scale and so amplify variance
    # well beyond the target. So instead, check the *relative* effect of `data=`:
    # samples should land closer to the target mean/scale than an otherwise-identical
    # flow without it, rather than matching the target outright.
    loc = jnp.array([5.0, -3.0, 1.0])
    scale = jnp.array([2.0, 0.1, 0.5])
    data = loc + scale * jr.normal(jr.key(6), (2048, 3))

    key = jr.key(7)
    with_data = _make_flow(flow_name, key, dim=3, data=data)
    without_data = _make_flow(flow_name, key, dim=3)

    sample_keys = jr.split(jr.key(8), 2048)
    samples_with = jax.vmap(with_data.sample)(sample_keys)
    samples_without = jax.vmap(without_data.sample)(sample_keys)

    mean_error_with = jnp.abs(samples_with.mean(axis=0) - loc).sum()
    mean_error_without = jnp.abs(samples_without.mean(axis=0) - loc).sum()
    assert mean_error_with < mean_error_without

    log_scale_error_with = jnp.abs(jnp.log(samples_with.std(axis=0)) - jnp.log(scale)).sum()
    log_scale_error_without = jnp.abs(
        jnp.log(samples_without.std(axis=0)) - jnp.log(scale)
    ).sum()
    assert log_scale_error_with < log_scale_error_without


@pytest.mark.parametrize("flow_name", ["coupling", "planar"])
def test_without_data_uses_reasonable_scale(flow_name):
    # masked_autoregressive is excluded here and checked separately below: under the
    # default invert=True, its `sample` uses a `dim`-step sequential inverse that
    # divides by an untrained, unconstrained scale at every step, so the *scale* of
    # an untrained MAF's samples is not reliably close to the base's -- unlike
    # coupling/planar, where sampling stays in the right ballpark even before
    # training. This is a loose bound, not a tight one: an untrained flow's exact
    # output scale still varies with the random init.
    flow = _make_flow(flow_name, jr.key(9), dim=2)
    samples = jax.vmap(flow.sample)(jr.split(jr.key(10), 2048))
    assert jnp.allclose(samples.std(axis=0), 1.0, atol=1.0)


def test_masked_autoregressive_untrained_samples_stay_finite():
    # See the comment on test_without_data_uses_standard_normal_scale: an untrained
    # MAF's samples can be far from unit scale, but they should never be NaN/inf --
    # the scale is bounded away from zero by `min_scale`, so the sequential inverse's
    # divisions stay well-defined however large the resulting values get.
    flow = fleqx.flows.masked_autoregressive_flow(jr.key(9), dim=2, flow_layers=2)
    samples = jax.vmap(flow.sample)(jr.split(jr.key(10), 256))
    assert jnp.all(jnp.isfinite(samples))


def test_planar_has_no_permutation_layer():
    # Unlike coupling/MAF, planar layers already depend on every dimension, so no
    # inter-layer permutation is needed.
    flow = fleqx.flows.planar_flow(jr.key(0), dim=DIM, flow_layers=3)
    assert "Permute" not in repr(flow.bijector)


def test_masked_autoregressive_works_for_larger_dim():
    # Exercises the scan-based inverse over more than a handful of steps.
    flow = fleqx.flows.masked_autoregressive_flow(jr.key(0), dim=20, flow_layers=2)
    x = flow.sample(jr.key(1))
    assert x.shape == (20,)
    assert jnp.isfinite(flow.log_prob(x))


def test_root_level_exports_match_submodule(flow_name):
    # fleqx.coupling_flow etc. are convenience aliases for the same objects, not
    # copies -- fleqx's whole API surface is flows, so these shouldn't need the
    # fleqx.flows.* prefix for everyday use.
    ctor, _ = FLOW_CONSTRUCTORS[flow_name]
    assert getattr(fleqx, ctor.__name__) is ctor
    assert fleqx.fit is fleqx.train.fit


@pytest.mark.skipif(
    _FORK_INSTALLED,
    reason="only exercises the fork-unavailable error path",
)
def test_template_raises_clear_error_without_fork(flow_name):
    ctor, kwargs = FLOW_CONSTRUCTORS[flow_name]
    template = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
    with pytest.raises(RuntimeError, match="gvcallen's fork"):
        ctor(jr.key(0), template=template, **kwargs)


@pytest.mark.skipif(not _FORK_INSTALLED, reason="requires gvcallen's distreqx fork")
class TestTemplateFlow:
    """`template=` lets a flow represent a distribution over an arbitrary PyTree."""

    TEMPLATE = {"a": jnp.zeros(2), "b": jnp.zeros((2, 2))}

    def _make(self, flow_name, key, **overrides):
        ctor, kwargs = FLOW_CONSTRUCTORS[flow_name]
        return ctor(key, template=self.TEMPLATE, flow_layers=2, **{**kwargs, **overrides})

    def test_sample_matches_template_structure(self, flow_name):
        flow = self._make(flow_name, jr.key(0))
        sample = flow.sample(jr.key(1))
        assert jax.tree_util.tree_structure(sample) == jax.tree_util.tree_structure(
            self.TEMPLATE
        )
        shapes = jax.tree_util.tree_map(lambda x: x.shape, sample)
        assert shapes == jax.tree_util.tree_map(lambda x: x.shape, self.TEMPLATE)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(sample))

    def test_log_prob_is_scalar(self, flow_name):
        flow = self._make(flow_name, jr.key(0))
        sample = flow.sample(jr.key(1))
        lp = flow.log_prob(sample)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_dim_is_inferred_as_total_template_size(self, flow_name):
        flow = self._make(flow_name, jr.key(0))
        assert flow.distribution.event_shape == (2 + 4,)

    def test_data_standardization_with_pytree_data(self, flow_name):
        data = {
            "a": jr.normal(jr.key(2), (256, 2)) * jnp.array([50.0, 0.02])
            + jnp.array([300.0, -20.0]),
            "b": jr.normal(jr.key(3), (256, 2, 2)) * 10.0 + 1000.0,
        }
        flow = self._make(flow_name, jr.key(4), data=data)
        sample = flow.sample(jr.key(5))
        assert jnp.isfinite(flow.log_prob(sample))

    def test_fit_reduces_loss(self, flow_name):
        # Plain standard-normal data (matching the untrained base distribution
        # exactly) gives training no clear signal to follow; shift/scale it like
        # the array-shaped flow tests do, for a robust target.
        data = {
            "a": jr.normal(jr.key(6), (512, 2)) * 0.5 + jnp.array([2.0, -1.0]),
            "b": jr.normal(jr.key(7), (512, 2, 2)) * 1.5 - 1.0,
        }
        flow = self._make(flow_name, jr.key(8))
        _, losses = fleqx.fit(
            jr.key(9),
            flow,
            data,
            learning_rate=1e-3,
            max_epochs=15,
            batch_size=64,
            show_progress=False,
        )
        assert all(jnp.isfinite(jnp.array(losses["train"])))
        assert losses["train"][-1] < losses["train"][0]
