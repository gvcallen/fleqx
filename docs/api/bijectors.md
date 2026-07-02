# Bijectors

Building blocks used by [the flow constructors](flows.md). Most users won't need
these directly, but they compose like any other
[`distreqx.bijectors.AbstractBijector`](https://lockwo.github.io/distreqx/api/bijectors/_bijector/)
if you want to assemble a flow by hand.

`Inverse` and `Permute` also exist in [gvcallen's distreqx
fork](https://github.com/gvcallen/distreqx) (pending upstream review); fleqx uses
that implementation when the fork is installed in place of the PyPI release of
`distreqx`, falling back to the bundled version documented here otherwise.

::: fleqx.bijectors.Coupling
    options:
        members:
            - __init__

---

::: fleqx.bijectors.MaskedAutoregressive
    options:
        members:
            - __init__

---

::: fleqx.bijectors.Planar
    options:
        members:
            - __init__

---

::: fleqx.bijectors.Permute
    options:
        members:
            - __init__

---

::: fleqx.bijectors.Inverse
    options:
        members: false
