"""Bijectors used to build normalizing flows.

Every class here is a genuine `distreqx.bijectors.AbstractBijector`, implemented
directly against the distreqx API with no third-party flow library involved. Most
users should build flows with a constructor from [`fleqx.flows`][] instead; this
module is for composing layers by hand.
"""

from ._coupling import Coupling as Coupling
from ._invert import Invert as Invert
from ._masked_autoregressive import MaskedAutoregressive as MaskedAutoregressive
from ._permute import Permute as Permute
from ._planar import Planar as Planar
