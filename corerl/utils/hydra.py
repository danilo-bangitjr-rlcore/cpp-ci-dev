import inspect
from dataclasses import field, fields, is_dataclass
from typing import Any, Concatenate, ParamSpec, TypeVar, Generic, Protocol
from collections.abc import Callable
from hydra.core.config_store import ConfigStore

T = TypeVar('T')
def list_(vals: list[T] | None = None) -> list[T]:
    if vals is None:
        return field(default_factory=list)

    return field(default_factory=lambda: vals.copy())


def interpolate(path: str) -> Any:
    return path


# -----------------------
# -- Group Dispatching --
# -----------------------

class DiscriminatedUnion(Protocol):
    name: str


Config = TypeVar('Config', bound=DiscriminatedUnion)
R = TypeVar('R')
P = ParamSpec('P')

class Group(Generic[R]):
    def __init__(
        self,
        group: str,
        return_type: type[R],
    ):
        self._group = group
        self._dispatchers: dict[str, Callable[..., R]] = {}


    def dispatcher(
        self,
        f: Callable[Concatenate[Config, P], R],
    ):
        args = inspect.getfullargspec(f)
        first = first_non_self_arg(args)
        config = args.annotations[first]
        name = get_config_name(config)

        cs = ConfigStore.instance()
        cs.store(name=name, group=self._group, node=config)
        self._dispatchers[name] = f

        return f

    def dispatch(self, config: DiscriminatedUnion, *args: Any):
        return self._dispatchers[config.name](config, *args)


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
