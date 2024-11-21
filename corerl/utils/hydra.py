import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Concatenate, TypeVar, Protocol
from collections.abc import Callable, Sequence
from hydra.core.config_store import ConfigStore

def list_(vals: list[Any] | None = None) -> Any:
    if vals is None:
        return field(default_factory=list)

    return field(default_factory=lambda: vals.copy())


def interpolate(path: str) -> Any:
    return path


def config(name: str, group: str | None = None):
    def _inner[T](cls: type[T]):
        node = dataclass(cls)

        cs = ConfigStore.instance()
        cs.store(name=name, group=group, node=node)

        return node

    return _inner


# -----------------------
# -- Group Dispatching --
# -----------------------

class DiscriminatedUnion(Protocol):
    name: str


Config = TypeVar('Config', bound=DiscriminatedUnion)

class Group[**P, R]:
    def __init__(
        self,
        group: str | Sequence[str],
    ):
        self._groups: Sequence[str] = [group] if isinstance(group, str) else group
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
        for group in self._groups:
            cs.store(name=name, group=group, node=config)

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
