import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Concatenate, Protocol, TypeVar

MISSING: Any = "|???|"

@dataclass
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

    def dispatcher(
        self,
        f: Callable[Concatenate[Config, P], R],
    ):
        args = inspect.getfullargspec(f)
        first = first_non_self_arg(args)
        config = args.annotations[first]
        name = get_config_name(config)

        self._dispatchers[name] = f

        return f

    def dispatch(self, config: DiscriminatedUnion, *args: P.args, **kwargs: P.kwargs):
        return self._dispatchers[config.name](config, *args, **kwargs)


def get_config_name(config: type[Any]):
    assert is_dataclass(config)
    config_fields = fields(config)
    name_field = next(filter(lambda n: n.name == 'name', config_fields))
    name = name_field.default
    assert isinstance(name, str)
    return name


def first_non_self_arg(args: inspect.FullArgSpec) -> str:
    for arg in args.args:
        if arg == 'self':
            continue

        return arg

    raise Exception('Failed to find non-self arg')

env_group = Group[[], Any]()

def init_env(cfg: EnvConfig):
    return env_group.dispatch(cfg)
