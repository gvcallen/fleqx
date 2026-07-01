"""Shared helper for bijectors parameterising an elementwise affine transform."""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array


def positive_scale(raw_scale: Array, min_scale: float) -> Array:
    """Maps an unconstrained array to strictly positive values via `softplus`.

    `min_scale` is added as a floor, both for numerical stability (avoiding
    `log(scale)` blowing up as `scale -> 0`) and to guarantee positivity outright.
    """
    return jnn.softplus(raw_scale) + min_scale


def identity_init_offset(min_scale: float) -> Array:
    """Offset added to a conditioner's raw scale output before `positive_scale`.

    A freshly initialised conditioner network outputs values close to zero, and
    `positive_scale(0, min_scale)` is around 0.7, not 1 -- so without this offset, a
    layer starts noticeably *shrinking* its input rather than close to the identity.
    That compounds across layers (e.g. 0.7^8 ~ 0.06), making a deep flow start far
    from the base distribution's scale and harder to train. Adding this offset before
    `positive_scale` makes `positive_scale(0 + offset, min_scale) == 1` exactly, so a
    freshly initialised layer starts at (approximately) the identity.
    """
    target = 1.0 - min_scale
    return jnp.log(-jnp.expm1(-target)) + target
