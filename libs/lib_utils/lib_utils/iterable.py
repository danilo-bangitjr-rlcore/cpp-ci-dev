from collections.abc import Hashable, Iterable, MutableMapping

from lib_utils.dict import keep


def partition[K: Hashable, V](
    iterable: Iterable[tuple[K, V]],
):
    result: dict[K, list[V]] = {}
    for key, value in iterable:
        result.setdefault(key, []).append(value)

    return result

def group_by[D: MutableMapping](
    iterable: Iterable[D],
):
    result: dict = {}
    for item in iterable:
        for key, value in item.items():
            result.setdefault(key, []).append(value)

    return result

def group_by_key[D: MutableMapping](
    iterable: Iterable[D],
    key: str,
    value: str,
):
    result: dict = {}
    for item in iterable:
        k = item[key]
        v = item[value]
        result.setdefault(k, []).append(v)

    return result

def keep_iterable[D: MutableMapping](
    iterable: Iterable[D],
    keys: list[str],
):
    return [keep(item, keys) for item in iterable]


