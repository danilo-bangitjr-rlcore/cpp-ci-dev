from collections.abc import Iterable
from typing import Any


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
