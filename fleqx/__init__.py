"""fleqx: normalizing flows for JAX, with a distreqx-native API."""

from . import bijectors as bijectors
from . import flows as flows
from . import train as train

try:  # pragma: no cover
    import importlib.metadata

    __version__ = importlib.metadata.version("fleqx")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.1"
