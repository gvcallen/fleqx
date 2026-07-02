"""Bijectors used to build normalizing flows.

Every class here is a genuine `distreqx.bijectors.AbstractBijector`, implemented
directly against the distreqx API with no third-party flow library involved. Most
users should build flows with a constructor from [`fleqx.flows`][] instead; this
module is for composing layers by hand.

`Coupling`, `MaskedAutoregressive` and `Planar` are fleqx-native: no equivalent
exists in `distreqx` itself (yet -- see [gvcallen's distreqx fork]
(https://github.com/gvcallen/distreqx) for bijectors pending upstream review).

`Inverse` and `Permute`, by contrast, also exist in that fork. fleqx depends only on
the PyPI release of `distreqx`, so a plain `pip install fleqx` works standalone; but
if the fork is installed in its place (`pip install
git+https://github.com/gvcallen/distreqx.git@main`), fleqx prefers its
implementations over the bundled fallbacks below. A bijector with no fallback should
follow the same `try`/`except ImportError` pattern, but raise a `RuntimeError` with
install instructions instead of falling back, since there's nothing to fall back to.
"""

try:
    from distreqx.bijectors import Inverse as Inverse
except ImportError:
    from ._inverse import Inverse as Inverse

try:
    from distreqx.bijectors import Permute as Permute
except ImportError:
    from ._permute import Permute as Permute

from ._coupling import Coupling as Coupling
from ._masked_autoregressive import MaskedAutoregressive as MaskedAutoregressive
from ._planar import Planar as Planar
