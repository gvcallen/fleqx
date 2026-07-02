"""fleqx: normalizing flows for JAX, with a distreqx-native API."""

from . import bijectors as bijectors
from . import flows as flows
from . import train as train
from .flows import (
    coupling_flow as coupling_flow,
    masked_autoregressive_flow as masked_autoregressive_flow,
    planar_flow as planar_flow,
)
from .train import fit as fit

try:  # pragma: no cover
    import importlib.metadata

    __version__ = importlib.metadata.version("fleqx")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.1"
