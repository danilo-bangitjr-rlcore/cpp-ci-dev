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


def post_processor[M](f: Callable[[Any, M], Any]):
    """"
    A post_processor is any arbitrary function that hooks onto the final stage of config
    processing. There are two common use cases:
      1. Sanity checking and early validation of config values that are beyond the scope of Pydantic's
         built-in tools.
      2. Mutating the config by filling in parts of the config tree based on other config values. For
         example, enabling configs based on feature flags.

    The post_processor receives the current (partially) validated config and a pointer to the top
    of the config tree (e.g. the MainConfig).
    """
    def _inner(self: Any, info: ValidationInfo):
        if info.context is None:
            return self

        try:
            f(self, info.context)
        except Exception as e:
            raise ValueError(str(e)) from e

        return self

    return model_validator(mode='after')(_inner)


def computed[M](key: str):
    """
    A config decorator for specifying config values that can be computed (or inferred)
    from other config values somewhere else in the config tree. This is executed after
    default values have been assigned and config group resolution has occurred.

    A computed config cannot mutate the current config instance. Instead, it must be a
    pure function of other config values in the tree. For example:
    ```python
    @computed('name')
    @classmethod
    def _some_meaningful_name(cls, cfg: 'MainConfig'):
        ...
    ```
    implies that the `name` attribute of this config can be computed by the given function
    of other configs.

    It is worth noting that the computed must be a `classmethod`. By definition, a computed
    config will cause recursive import errors unless the top of the tree is imported with
    a type guard.
    ```python
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from corerl.config import MainConfig
    ```
    """
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
            try:
                data[key] = f.__get__(None, cls)(info.context)
            except Exception as e:
                raise ValueError() from e

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
