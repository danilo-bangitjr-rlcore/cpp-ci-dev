from collections.abc import Callable
from dataclasses import replace
from typing import Any, Concatenate, Protocol, TypeVar

from lib_config.config import config

MISSING: Any = "|???|"

@config(frozen=True)
class EnvConfig:
    name : str = MISSING
    seed : int = 0

class DiscriminatedUnion(Protocol):
    @property
    def name(self) -> Any: ...

Config = TypeVar('Config', bound=DiscriminatedUnion)

class Group[**P, R]:
    def __init__(self):
        self._dispatchers: dict[str, Callable[..., R]] = {}
        self._cfgs: dict[str, EnvConfig] = {}

    def dispatcher(
        self,
        cfg: EnvConfig,
        f: Callable[Concatenate[Config, P], R],
    ):
        self._dispatchers[cfg.name] = f
        self._cfgs[cfg.name] = cfg
        return f

    def dispatch(self, name: str, overrides: dict | None = None, cfg_obj: Any = None):
        if cfg_obj is not None:
            return self._dispatchers[name](cfg_obj)
        if overrides is None:
            overrides = {}
        cfg = self._cfgs[name]
        cfg = replace(cfg, **overrides)
        return self._dispatchers[name](cfg)

env_group = Group[[], Any]()
