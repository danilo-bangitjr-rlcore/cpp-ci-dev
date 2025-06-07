from collections.abc import Callable, Sequence
from inspect import Parameter, signature

import jax


def jit[**P, R](f: Callable[P, R], static_argnums: tuple[int, ...] | None = None) -> Callable[P, R]:
    return jax.jit(f, static_argnums=static_argnums)


def method_jit[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    return jax.jit(f, static_argnums=(0,))


def vmap[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    return jax.vmap(f)


def vmap_except[F: Callable](f: F, exclude: Sequence[str | int], levels: int = 1) -> F:
    """
    vmap over all arguments except those in `exclude`.

    ```python
    @partial(vmap_except, exclude=["x"])
    def f(x: jax.Array, y: jax.Array)
        return x + y
    ```
    """
    if levels == 0:
        return f

    sig = signature(f)
    args = [
        k for k, p in sig.parameters.items() if p.default is Parameter.empty
    ]

    total: list[int | None] = [0] * len(args)
    for i, k in enumerate(args):
        if k in exclude or i in exclude:
            total[i] = None

    for _ in range(levels):
        f = jax.vmap(f, in_axes=total)

    return f

def vmap_only[F: Callable](f: F, include: Sequence[str | int], levels: int = 1) -> F:
    """
    vmap over only the arguments specified in `include`.

    ```python
    @partial(vmap_only, include=["x"])
    def f(x: jax.Array, y: jax.Array)
        return x + y
    ```
    """
    if levels == 0:
        return f

    sig = signature(f)
    args = [
        k for k, p in sig.parameters.items() if p.default is Parameter.empty
    ]

    total: list[int | None] = [None] * len(args)
    for i, k in enumerate(args):
        if k in include or i in include:
            total[i] = 0

    for _ in range(levels):
        f = jax.vmap(f, in_axes=total)

    return f


def multi_vmap[F: Callable](f: F, levels: int) -> F:
    for _ in range(levels):
        f = jax.vmap(f)

    return f
