import copy
from dataclasses import field
from typing import Any, dataclass_transform

from pydantic import Field, PrivateAttr
from pydantic.dataclasses import dataclass as pydantic_dataclass

MISSING: Any = "|???|"

def list_(vals: list[Any] | None = None) -> Any:
    if vals is None:
        return field(default_factory=list)

    return field(default_factory=lambda: copy.deepcopy(vals))


def interpolate(path: str) -> Any:
    return path


@dataclass_transform(field_specifiers=(field, Field, PrivateAttr))
def config(
    *,
    frozen: bool = False,
    allow_extra: bool = False,
):
    def _inner(cls: Any):
        return pydantic_dataclass(
            cls,
            frozen=frozen,
            config={
                'extra': 'ignore' if allow_extra else 'forbid',
            },
        )
    return _inner
