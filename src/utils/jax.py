from collections.abc import Callable

import jax


def jit[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    return jax.jit(f)


def method_jit[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    return jax.jit(f, static_argnums=(0,))
