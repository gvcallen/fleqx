"""Support for bijectors only available in gvcallen's distreqx fork.

fleqx depends only on the PyPI release of `distreqx`, so a plain `pip install
fleqx` works standalone. A handful of bijectors used here also exist in
[gvcallen's distreqx fork](https://github.com/gvcallen/distreqx) -- currently a
set of PRs pending upstream review -- and fleqx prefers them when that fork is
installed in place of the PyPI release. `Permute` and `Inverse` (see
`fleqx.bijectors`) have a bundled fallback for when it isn't; bijectors with no
such fallback should use `require` below instead, which raises a clear error.
"""

from __future__ import annotations

import importlib
from typing import Any


def require(name: str, *, needed_by: str) -> Any:
    """Returns `distreqx.bijectors.<name>`, or raises if it isn't available.

    **Arguments:**

    - `name`: Attribute name to look up in `distreqx.bijectors`.
    - `needed_by`: Human-readable description of what needs it, used in the error
        message (e.g. a flow constructor's name).
    """
    bijector = getattr(importlib.import_module("distreqx.bijectors"), name, None)
    if bijector is None:
        raise RuntimeError(
            f"{needed_by} requires `distreqx.bijectors.{name}`, which isn't in the "
            "PyPI release of distreqx. Install gvcallen's fork instead: "
            "`pip install git+https://github.com/gvcallen/distreqx.git@main`."
        )
    return bijector
