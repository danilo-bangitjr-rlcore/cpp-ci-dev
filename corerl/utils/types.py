from __future__ import annotations
from typing import Any, Protocol


class Ring(Protocol):
    def __add__(self, other: Any, /) -> Any:
        ...

    def __sub__(self, other: Any, /) -> Any:
        ...

    def __mul__(self, other: Any, /) -> Any:
        ...

    def __rmul__(self, other: Any, /) -> Any:
        ...

    def __pow__(self, other: Any, /) -> Any:
        ...
