import sys
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Concatenate

import yaml
from pydantic import TypeAdapter
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic.fields import FieldInfo

import corerl.utils.dict as dict_u
from corerl.utils.list import find
from corerl.utils.maybe import Maybe


# -------------------
# -- CLI Overrides --
# -------------------
def _get_equals_or_next(flag: str, idx: int):
    """
    Two valid command line syntax:
      --flag=val
      --flag val

    The first is parsed by `sys.argv` as a single entry
    that needs to be split. The second is parsed as two
    adjacent entries.

    This function returns a {flag: val} dict if either
    syntax is used, or an empty {} if neither is detected.
    """
    chunk = sys.argv[idx]

    if not chunk.startswith(f'--{flag}'):
        return {}

    chunk = chunk.removeprefix(f'--{flag}')
    if chunk.startswith('='):
        return { flag: chunk.removeprefix('=') }

    return { flag: sys.argv[idx + 1] }


def _get_anonymous_equals(idx: int):
    """
    Parse any cli flags that are not prefixed
    by two hyphens (--) and that contain an
    equal sign (=).
      flag=val
      a.b.c=val

    Ignore flags of the form
      --flag=val
      --flag val
      flag val
    """
    chunk = sys.argv[idx]

    if chunk.startswith('--'):
        return {}

    if '=' not in chunk:
        return {}

    name, val = chunk.split('=')
    return { name: val }


def _flags_from_cli():
    flags: dict[str, str] = {}
    for i in range(1, len(sys.argv)):
        flags = (
            flags
            | _get_equals_or_next('base', i)
            | _get_equals_or_next('config-name', i)
            | _get_anonymous_equals(i)
        )

    return flags


# --------------------------
# -- YAML Default Merging --
# --------------------------

# NOTE: this takes a pydantic.fields.Field, however the public type interface
# is incomplete, causing pyright to raise an error. In leiu of better types,
# treating this as the most restrictive type and guarding all accessors.
def _get_discriminator(field: object) -> str | None:
    """
    A discriminated union can be constructed in two different ways,
    each producing a slightly different object hierarchy.
    1. Field(discriminator='name') as a dataclass default value
    2. Annotated[..., Field(discriminator='name')] when defining a union type
    """
    # first look directly at the Field attributes to see if it is a discriminator
    field_disc = (
        Maybe(getattr(field, 'default', None))
        .map(lambda d: getattr(d, 'discriminator', None))
    )

    # otherwise walk through the type annotation and ask if it has a discriminator
    type_disc = (
        Maybe(getattr(field, 'default', None))
        .map(lambda d: getattr(d, 'metadata', None))
        .map(lambda meta: find(lambda f: f.discriminator, meta))
        .map(lambda meta: meta.discriminator)
    )

    return (
        field_disc
        .flat_otherwise(lambda: type_disc)
        .unwrap()
    )

def _resolve_default_discriminated_unions(config: Any) -> dict[str, object]:
    """
    Recursively walk through the **default** values of a config type
    and resolve any discriminated unions.

    Pydantic cannot resolve discriminated unions that have a default value.
    For example, the following config will fail if relying only on default values:
    ```
    @config()
    class SomeConfig:
      actor: ActorA | ActorB = Field(default_factory=ActorA, discriminator='name')
    ```
    because Pydantic will complain that there is no "name" field to use for resolving
    the discriminator.

    By first resolving all default discriminator values, we can sprinkle the defaults
    into the raw config --- overwritten by nondefault values --- to prevent pydantic
    from complaining.
    """
    out = {}

    if not is_pydantic_dataclass(config) and not is_dataclass(config):
        return out

    for v in fields(config):
        # if there is no default value, we know this cannot
        # be an object, so we don't need to walk down this path
        if not isinstance(v.default, FieldInfo):
            continue

        discriminator = _get_discriminator(v)

        # if there is a default object, we need to walk it for
        # any sub-discriminated unions
        if v.default.default_factory is None:
            continue

        factory: Any = v.default.default_factory
        sub_cfg = factory()

        if isinstance(sub_cfg, list):
            continue

        out[v.name] = {}
        if discriminator is not None:
            out[v.name][discriminator] = getattr(sub_cfg, discriminator)

        out[v.name] |= _resolve_default_discriminated_unions(sub_cfg)

        # if the sub-tree does not contain any discriminated unions
        # then just remove it from the defaults tree
        if len(out[v.name]) == 0:
            del out[v.name]

    return out

def _load_raw_config(base: str, config_name: str) -> dict[str, Any]:
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'

    path = Path(base) / f'{config_name}'
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # if a config does not load subfiles
    # then our work here is done
    if 'defaults' not in config:
        return config

    # otherwise recursively load each subfile
    # and merge using the parent as precedent
    for default in config['defaults']:
        # if the subfile path is just a string,
        # then merge with the entire parent
        if isinstance(default, str):
            def_config = _load_raw_config(base, default)
            config = dict_u.merge(def_config, config)

        # if the subfile is a key-value pair,
        # the key specifies the sub-dictionary to
        # merge the subfile into.
        elif isinstance(default, dict):
            k = list(default.keys())[0]
            v = list(default.values())[0]

            if not dict_u.has_path(config, k):
                config = dict_u.set_at_path(config, k, {})

            raw_def_config = _load_raw_config(base, v)
            def_config = dict_u.set_at_path({}, k, raw_def_config)
            config = dict_u.merge(def_config, config)

    del config['defaults']
    return config


def direct_load_config[T](
    Config: type[T],
    overrides: dict[str, str] | None = None,
    base: str | None = None,
    config_name: str | None = None,
):
    # parse all of the command line flags
    # gracefully ignore those we can't parse
    flags = _flags_from_cli()
    flags |= (overrides or {})

    # give precedence to cli overrides
    # else, require function args to be specified
    base = flags.get('base', base)
    config_name = flags.get('config-name', config_name)
    assert base is not None and config_name is not None, 'Must specify a base path for configs and a config name'

    # remove the `base` from the config_name if it already exists
    if config_name.startswith(base):
        config_name = config_name[len(base):]

    # load the raw config with defaults resolved
    raw_config = _load_raw_config(base, config_name)
    return config_from_dict(Config, raw_config, flags=flags)


def config_from_dict[T](Config: type[T], raw_config: dict, flags: dict[str, str] | None = None):
    raw_config = dict_u.merge(
        _resolve_default_discriminated_unions(Config),
        raw_config,
    )

    # we are not supporting config defaults when loading from dict
    if "defaults" in raw_config.keys():
        del raw_config['defaults']

    # handle any cli overrides
    if flags is not None:
        cli_overrides = dict_u.drop(flags, ['base', 'config-name'])
        for override_key, override_value in cli_overrides.items():
            dict_u.set_at_path(raw_config, override_key, override_value)

    # validate config against provided schema, Config.
    # raise exception on extra values not in schema
    ta = TypeAdapter(Config)

    # first validation pass to resolve discriminated unions
    obj_config: Any = ta.validate_python(raw_config)

    # second validation pass to execute pre/post computed hooks
    output_config = config_to_dict(Config, obj_config)
    return ta.validate_python(output_config, context=obj_config)

# ----------------
# -- Public API --
# ----------------
def load_config[T](Config: type[T], base: str | None = None, config_name: str | None = None):
    def _inner[**U, R](f: Callable[Concatenate[T, U], R]):
        def __inner(*args: U.args, **kwargs: U.kwargs) -> R:
            config = direct_load_config(Config, base=base, config_name=config_name)
            return f(config, *args, **kwargs)
        return __inner
    return _inner


def config_to_dict(Config: type[object], config: object):
    ta = TypeAdapter(Config)
    return ta.dump_python(config, warnings=False)

def config_to_json(Config: type[object], config: object):
    ta = TypeAdapter(Config)
    return ta.dump_json(config, warnings=False)
