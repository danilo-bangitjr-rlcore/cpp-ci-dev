from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar
from annotated_types import SupportsLt, SupportsGt


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
