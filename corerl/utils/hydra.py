from dataclasses import field
from typing import TypeVar

T = TypeVar('T')
def list_(vals: list[T] | None = None) -> list[T]:
    if vals is None:
        return field(default_factory=list)

    return field(default_factory=lambda: vals.copy())
