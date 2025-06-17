from collections.abc import Callable, Sequence
from inspect import Parameter, signature
from typing import Any, Literal, overload

try:
    import jax
except ImportError as e:
    raise ImportError(
        "JAX is not installed. Please install it with `uv add --group=jax`.",
    ) from e


def jit[**P, R](f: Callable[P, R], static_argnums: tuple[int, ...] | None = None) -> Callable[P, R]:
    return jax.jit(f, static_argnums=static_argnums)


def method_jit[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    return jax.jit(f, static_argnums=(0,))


@overload
def grad[**P](f: Callable[P, jax.Array], has_aux: Literal[False] = False) -> Callable[P, jax.Array]: ...
@overload
def grad[**P, *R](
    f: Callable[P, tuple[jax.Array, *R]],
    has_aux: Literal[True] = True,
) -> Callable[P, tuple[jax.Array, *R]]: ...

def grad[F: Callable](f: F, has_aux: bool = False) -> F:
    g: Any = jax.grad(f, has_aux=has_aux)
    return g


def vmap[F: Callable](f: F, in_axes: tuple[int | None, ...] | None = None) -> F:
    # if no in_axes are provided, we assume all arguments are batched
    if in_axes is None:
        in_axes = tuple(
            0 if p.default is Parameter.empty else None
            for p in signature(f).parameters.values()
        )

    return jax.vmap(f, in_axes=in_axes)


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
        f = jax.vmap(f, in_axes=tuple(total))

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
        f = jax.vmap(f, in_axes=tuple(total))

    return f


def multi_vmap[F: Callable](f: F, levels: int) -> F:
    for _ in range(levels):
        f = jax.vmap(f)

    return f
