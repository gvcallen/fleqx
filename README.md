# fleqx

Normalizing flows for JAX, built directly on [distreqx](https://github.com/lockwo/distreqx).

A flow from `fleqx` is a plain `distreqx.distributions.Transformed` — there's no
flow-specific wrapper, so `log_prob`, `sample`, `optax` training, etc. all work
exactly as they would for any other distreqx distribution. As with any distreqx
distribution, batches are handled with `jax.vmap` rather than by passing in arrays
with a leading batch axis.

- `fleqx.train.fit` accepts an arbitrary pytree for both `dist` and `data` -- e.g. a
  dict of independently-trained distributions and their corresponding data arrays --
  not just a single distribution and a single `(n, dim)` array. Pytree leaves may be
  `None` for fields you're not using.
- Each flow constructor also accepts `template=` in place of `dim`, for a
  distribution over an arbitrary pytree of arrays (e.g. a dict of named variables)
  instead of a flat vector. Requires gvcallen's distreqx fork (see Installation).
- Bijectors that also exist in [gvcallen's distreqx
  fork](https://github.com/gvcallen/distreqx) are used automatically when that fork
  is installed in place of the PyPI release of `distreqx` -- see Installation below.

Three flow types are implemented so far: coupling, masked autoregressive, and planar.

## Example

```python
import jax.numpy as jnp
import jax.random as jr

import fleqx

key = jr.key(0)
flow = fleqx.coupling_flow(key, dim=2)

sample = flow.sample(jr.key(1))
log_p = flow.log_prob(sample)

data = jr.normal(jr.key(2), (1000, 2))
flow, losses = fleqx.fit(jr.key(3), flow, data)
```

`coupling_flow`, `masked_autoregressive_flow`, `planar_flow` and `fit` are also
available at their fully-qualified paths (`fleqx.flows.coupling_flow`,
`fleqx.train.fit`, etc.), alongside the rest of each submodule.

## Installation

```
pip install fleqx
```

fleqx depends only on the PyPI release of
[distreqx](https://github.com/lockwo/distreqx), so this is enough on its own. We
recommend also installing [gvcallen's distreqx
fork](https://github.com/gvcallen/distreqx) in its place, though: it includes
several additional bijectors (pending upstream review as PRs) that fleqx will prefer
automatically over its own bundled fallbacks, with no fleqx-side configuration
needed.

```
pip install git+https://github.com/gvcallen/distreqx.git@main
```

See the [documentation](https://gvcallen.github.io/fleqx) for the full API.

## Acknowledgements

Built on [distreqx](https://github.com/lockwo/distreqx) (Owen Lockwood) and
[parax](https://github.com/gvcallen/parax) (freezing/unwrapping of fixed
sub-components, e.g. `data=`'s standardizing layer). The bijectors were ported from
[flowjax](https://github.com/danielward27/flowjax) (Daniel Ward), a more complete
flows library that's worth using directly if you don't specifically need the
distreqx API.

## Development

```
pip install -e ".[test]"
pytest
```

---

This library was written by Claude, porting the bijectors directly from flowjax.
Behaviour should be nearly identical to flowjax's, though minor differences may
remain, and the code hasn't yet had a full human review.
