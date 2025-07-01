from collections.abc import Callable

BoundsElem = float | str | None

Bounds = tuple[BoundsElem, BoundsElem]
FloatBounds = tuple[float | None, float | None]

BoundsFunction = tuple[Callable[..., float] | None, Callable[..., float] | None]
BoundsTags = tuple[list[str] | None, list[str] | None]
