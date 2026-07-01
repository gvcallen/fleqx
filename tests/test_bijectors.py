"""Tests for the native distreqx bijectors backing fleqx's coupling flow."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from distreqx.bijectors import AbstractBijector

from fleqx.bijectors import Coupling, Invert, MaskedAutoregressive, Permute, Planar

DIM = 5


@pytest.fixture
def coupling():
    return Coupling(
        jr.key(0), untransformed_dim=2, dim=DIM, nn_width=16, nn_depth=1
    )


@pytest.fixture
def maf():
    return MaskedAutoregressive(jr.key(0), dim=DIM, nn_width=16, nn_depth=1)


@pytest.fixture
def planar():
    return Planar(jr.key(0), dim=DIM, negative_slope=0.1)


def _numerical_log_det(fn, x):
    jac = jax.jacfwd(fn)(x)
    _, log_det = jnp.linalg.slogdet(jac)
    return log_det


class TestCoupling:
    def test_is_bijector(self, coupling):
        assert isinstance(coupling, AbstractBijector)
        assert coupling.is_constant_jacobian is False
        assert coupling.is_constant_log_det is False

    def test_untransformed_part_passes_through(self, coupling):
        x = jr.normal(jr.key(1), (DIM,))
        y = coupling.forward(x)
        assert jnp.array_equal(y[:2], x[:2])

    def test_round_trip(self, coupling):
        x = jr.normal(jr.key(1), (DIM,))
        y, fwd_log_det = coupling.forward_and_log_det(x)
        x_recovered, inv_log_det = coupling.inverse_and_log_det(y)
        assert jnp.allclose(x, x_recovered, atol=1e-5)
        assert jnp.allclose(fwd_log_det, -inv_log_det, atol=1e-5)

    def test_log_det_matches_autodiff_jacobian(self, coupling):
        x = jr.normal(jr.key(1), (DIM,))
        _, log_det = coupling.forward_and_log_det(x)
        numerical = _numerical_log_det(coupling.forward, x)
        assert jnp.allclose(log_det, numerical, atol=1e-4)

    def test_scale_is_always_positive(self):
        # A large negative conditioner output should still map to a positive scale,
        # via the softplus + min_scale reparameterization.
        coupling = Coupling(
            jr.key(2), untransformed_dim=2, dim=DIM, nn_width=16, nn_depth=1
        )
        x_cond = jnp.full((2,), -1e4)
        _, scale = coupling._shift_and_scale(x_cond)
        assert jnp.all(scale > 0)

    def test_different_keys_give_different_transforms(self):
        x = jr.normal(jr.key(1), (DIM,))
        c0 = Coupling(jr.key(0), untransformed_dim=2, dim=DIM, nn_width=16, nn_depth=1)
        c1 = Coupling(jr.key(1), untransformed_dim=2, dim=DIM, nn_width=16, nn_depth=1)
        assert not jnp.allclose(c0.forward(x), c1.forward(x))


class TestPermute:
    def test_is_bijector_with_constant_jacobian(self):
        perm = Permute(jnp.array([2, 0, 1, 4, 3]))
        assert isinstance(perm, AbstractBijector)
        assert perm.is_constant_jacobian is True
        assert perm.is_constant_log_det is True

    def test_forward_reorders_as_specified(self):
        perm = Permute(jnp.array([2, 0, 1]))
        x = jnp.array([10.0, 20.0, 30.0])
        y = perm.forward(x)
        assert jnp.array_equal(y, jnp.array([30.0, 10.0, 20.0]))

    def test_round_trip(self):
        perm = Permute(jr.permutation(jr.key(0), DIM))
        x = jr.normal(jr.key(1), (DIM,))
        y, fwd_log_det = perm.forward_and_log_det(x)
        x_recovered, inv_log_det = perm.inverse_and_log_det(y)
        assert jnp.array_equal(x, x_recovered)
        assert fwd_log_det == 0.0
        assert inv_log_det == 0.0


class TestMaskedAutoregressive:
    def test_is_bijector(self, maf):
        assert isinstance(maf, AbstractBijector)
        assert maf.is_constant_jacobian is False
        assert maf.is_constant_log_det is False

    def test_round_trip(self, maf):
        x = jr.normal(jr.key(1), (DIM,))
        y, fwd_log_det = maf.forward_and_log_det(x)
        x_recovered, inv_log_det = maf.inverse_and_log_det(y)
        assert jnp.allclose(x, x_recovered, atol=1e-4)
        assert jnp.allclose(fwd_log_det, -inv_log_det, atol=1e-4)

    def test_log_det_matches_autodiff_jacobian(self, maf):
        x = jr.normal(jr.key(1), (DIM,))
        _, log_det = maf.forward_and_log_det(x)
        numerical = _numerical_log_det(maf.forward, x)
        assert jnp.allclose(log_det, numerical, atol=1e-4)

    def test_jacobian_is_lower_triangular(self, maf):
        # The defining property of an autoregressive transform: y[i] depends only
        # on x[:i+1], so dy[i]/dx[j] == 0 for j > i.
        x = jr.normal(jr.key(1), (DIM,))
        jac = jax.jacfwd(maf.forward)(x)
        assert jnp.allclose(jnp.triu(jac, k=1), 0.0, atol=1e-6)

    def test_dim_one_is_unconditional(self):
        # With nothing earlier to condition on, dimension 0's transform must be a
        # fixed (input-independent) affine map: changing x should not change the
        # shift/scale the conditioner produces.
        maf = MaskedAutoregressive(jr.key(0), dim=1, nn_width=8, nn_depth=1)
        shift0, scale0 = maf._shift_and_scale(maf._conditioner(jnp.array([0.0])))
        shift1, scale1 = maf._shift_and_scale(maf._conditioner(jnp.array([5.0])))
        assert jnp.allclose(shift0, shift1)
        assert jnp.allclose(scale0, scale1)

    def test_different_keys_give_different_transforms(self):
        x = jr.normal(jr.key(1), (DIM,))
        m0 = MaskedAutoregressive(jr.key(0), dim=DIM, nn_width=16, nn_depth=1)
        m1 = MaskedAutoregressive(jr.key(1), dim=DIM, nn_width=16, nn_depth=1)
        assert not jnp.allclose(m0.forward(x), m1.forward(x))


class TestPlanar:
    def test_is_bijector(self, planar):
        assert isinstance(planar, AbstractBijector)
        assert planar.is_constant_jacobian is False
        assert planar.is_constant_log_det is False

    def test_round_trip(self, planar):
        x = jr.normal(jr.key(1), (DIM,))
        y, fwd_log_det = planar.forward_and_log_det(x)
        x_recovered, inv_log_det = planar.inverse_and_log_det(y)
        assert jnp.allclose(x, x_recovered, atol=1e-4)
        assert jnp.allclose(fwd_log_det, -inv_log_det, atol=1e-4)

    def test_log_det_matches_autodiff_jacobian(self, planar):
        x = jr.normal(jr.key(1), (DIM,))
        _, log_det = planar.forward_and_log_det(x)
        numerical = _numerical_log_det(planar.forward, x)
        assert jnp.allclose(log_det, numerical, atol=1e-4)

    def test_invalid_negative_slope_raises(self):
        with pytest.raises(ValueError):
            Planar(jr.key(0), dim=DIM, negative_slope=0.0)
        with pytest.raises(ValueError):
            Planar(jr.key(0), dim=DIM, negative_slope=1.0)

    def test_different_keys_give_different_transforms(self):
        x = jr.normal(jr.key(1), (DIM,))
        p0 = Planar(jr.key(0), dim=DIM, negative_slope=0.1)
        p1 = Planar(jr.key(1), dim=DIM, negative_slope=0.1)
        assert not jnp.allclose(p0.forward(x), p1.forward(x))


class TestInvert:
    def test_swaps_forward_and_inverse(self, coupling):
        inverted = Invert(coupling)
        x = jr.normal(jr.key(1), (DIM,))
        assert jnp.array_equal(inverted.forward(x), coupling.inverse(x))
        assert jnp.array_equal(inverted.inverse(x), coupling.forward(x))

    def test_swaps_log_det(self, coupling):
        inverted = Invert(coupling)
        x = jr.normal(jr.key(1), (DIM,))
        _, inverted_fwd_log_det = inverted.forward_and_log_det(x)
        _, coupling_inv_log_det = coupling.inverse_and_log_det(x)
        assert jnp.allclose(inverted_fwd_log_det, coupling_inv_log_det)

    def test_double_invert_matches_original(self, coupling):
        double_inverted = Invert(Invert(coupling))
        x = jr.normal(jr.key(1), (DIM,))
        assert jnp.array_equal(double_inverted.forward(x), coupling.forward(x))

    def test_inherits_constant_jacobian_flag(self):
        perm = Permute(jnp.array([1, 0]))
        assert Invert(perm).is_constant_jacobian is True
        assert Invert(perm).is_constant_log_det is True
