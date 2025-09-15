from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

T_contra = TypeVar("T_contra", contravariant=True)

@runtime_checkable
class SqlWriter(Protocol[T_contra]):
    """Protocol for writing sequences of row-like objects to a SQL backend.

    Minimal surface used by higher-level buffering wrappers.
    Implementations should be thread-safe iff they are expected to be wrapped
    by a buffered writer using a background thread.
    """

    def write_many(self, rows: Sequence[T_contra]) -> None:
        ...

    def write(self, row: T_contra) -> None:
        ...

    def close(self) -> None:
        ...
