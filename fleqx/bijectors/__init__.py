"""Bijectors used to build normalizing flows.

Every class here is a genuine `distreqx.bijectors.AbstractBijector`, implemented
directly against the distreqx API with no third-party flow library involved. Most
users should build flows with a constructor from [`fleqx.flows`][] instead; this
module is for composing layers by hand.

`Coupling`, `MaskedAutoregressive` and `Planar` are fleqx-native: no equivalent
exists in `distreqx` itself. `Inverse` and `Permute` come from
[`parax.bijectors`](https://gvcallen.github.io/parax), which re-exports the
installed `distreqx`'s versions where it has them and fills in the rest, so any
`distreqx` from the PyPI release onwards works.
"""

from parax.bijectors import (
    Inverse as Inverse,
    Permute as Permute,
)

from ._coupling import Coupling as Coupling
from ._masked_autoregressive import MaskedAutoregressive as MaskedAutoregressive
from ._planar import Planar as Planar
