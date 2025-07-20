from collections.abc import Hashable, Iterable


def partition[K: Hashable, V](
    iterable: Iterable[tuple[K, V]],
):
    result: dict[K, list[V]] = {}
    for key, value in iterable:
        result.setdefault(key, []).append(value)

    return result
