from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from annotated_types import SupportsGt, SupportsLt


def find[T](pred: Callable[[T], bool], li: Iterable[T]) -> T | None:
    for item in li:
        if pred(item):
            return item


def find_index[T](pred: Callable[[T], bool], li: Iterable[T]) -> int | None:
    for i, item in enumerate(li):
        if pred(item):
            return i


def find_instance[T, U](inst: type[U], li: Iterable[T]) -> U | None:
    item = find(lambda x: isinstance(x, inst), li)
    if item and isinstance(item, inst):
        return item


def flatten(li: Iterable[Any]) -> list[Any]:
    out = []

    def _inner(item: Any):
        if isinstance(item, Iterable):
            for sub in item:
                _inner(sub)

        else:
            out.append(item)

    _inner(li)
    return out


def partition[T](pred: Callable[[T], bool], li: Iterable[T]) -> tuple[list[T], list[T]]:
    left = []
    right = []

    for item in li:
        if pred(item):
            left.append(item)
        else:
            right.append(item)

    return left, right


T = TypeVar('T', bound=SupportsLt | SupportsGt)
def multi_level_sort(
    vals: Sequence[T],
    categories: list[Callable[[T], bool]],
):
    rest = vals
    sorted_vals: list[T] = []

    for cat in categories:
        left, right = partition(cat, rest)
        sorted_vals += sorted(left)
        rest = right

    sorted_vals += sorted(rest)
    return sorted_vals


def sort_by(a: list[Any], b:list[Any]):
    """
    sorts a and b by b, then returns both lists sorted
    """
    paired = list(zip(a, b, strict=True))
    sorted_pairs = sorted(paired, key=lambda pair: pair[1])
    sorted_a, sorted_b = zip(*sorted_pairs, strict=True)
    return list(sorted_a), list(sorted_b)
