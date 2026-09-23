# Bijectors

Building blocks used by [the flow constructors](flows.md). Most users won't need
these directly, but they compose like any other
[`distreqx.bijectors.AbstractBijector`](https://lockwo.github.io/distreqx/api/bijectors/_bijector/)
if you want to assemble a flow by hand.

`fleqx.bijectors` also re-exports `Inverse` and `Permute` from
[`parax.bijectors`](https://gvcallen.github.io/parax), which takes them from the
installed `distreqx` where it has them and fills them in where it doesn't. They are
documented there rather than here, since which implementation you get depends on
your `distreqx`.

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
