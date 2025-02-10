import copy
from collections.abc import Callable
from dataclasses import field
from typing import Any, dataclass_transform

from pydantic import Field, PrivateAttr, ValidationInfo, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

MISSING: Any = "|???|"

def list_(vals: list[Any] | None = None) -> Any:
    if vals is None:
        return Field(default_factory=list)

    return Field(default_factory=lambda: copy.deepcopy(vals))


def interpolate(path: str) -> Any:
    return path


def sanitizer[M](f: Callable[[Any, M], Any]):
    def _inner(self: Any, info: ValidationInfo):
        if info.context is None:
            return self

        f(self, info.context)
        return self

    return model_validator(mode='after')(_inner)


def computed[M](key: str):
    def _computed_processor(f: Callable[[type[Any], M], Any]):
        def _model_validator_callback(cls: Any, data: object, info: ValidationInfo):
            if info.context is None:
                return data

            assert isinstance(data, dict)

            # if the value is already specified in the data
            # then we've overridden the computed logic
            if key in data and data[key] != MISSING:
                return data

            # because f is _actually_ a descriptor instead of a function
            # due to classmethod, then we need to get to the underlying
            # function using the descriptor api
            data[key] = f.__get__(None, cls)(info.context)
            return data

        return model_validator(mode='before')(_model_validator_callback)
    return _computed_processor



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
