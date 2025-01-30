from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import _MISSING_TYPE, fields, is_dataclass
from inspect import isclass
from typing import Any

from pydantic.fields import FieldInfo

import corerl.utils.nullable as nullable


def assign_default[K, V](d: dict[K, V], key: K, default: Callable[[], V]) -> V:
    if key in d:
        return d[key]

    out = d[key] = default()
    return out


def drop(d: MutableMapping, to_drop: Sequence[str]) -> dict:
    to_keep = set(d.keys()) - set(to_drop)
    return {
        k: d[k] for k in to_keep
     }


def flatten(
    d: MutableMapping,
    separator: str = "_",
    _parent_key: str = "",
    _carry: dict | None = None,
) -> dict:
    _carry = _carry if _carry is not None else {}

    for k, v in d.items():
        new_k = _parent_key + k
        if isinstance(v, MutableMapping):
            new_parent = new_k + separator
            flatten(v, separator, new_parent, _carry)
        else:
            _carry[new_k] = v

    return _carry


def hash(
    d: MutableMapping[str, Any],
    ignore: Iterable[str] | None = None,
    _parent_path: str = '',
    _hash: hashlib._Hash | None = None,
) -> str:
    ignore = nullable.default(ignore, set)
    _hash = nullable.default(_hash, hashlib.sha1)

    for k in sorted(d.keys()):
        path = _parent_path + k

        if path in ignore:
            continue

        if isinstance(d[k], MutableMapping):
            hash(
                d[k],
                ignore,
                path + '.',
                _hash,
            )

        else:
            _hash.update(str(d[k]).encode('utf-8'))

    return _hash.hexdigest()


def hash_many(
    ds: Iterable[MutableMapping[str, Any]],
    ignore: Iterable[str] | None = None,
) -> str:
    # build a shared hasher that is reused for each
    # dictionary in the list to ensure a single
    # consistent hash is produced
    hasher = hashlib.sha1()

    for d in ds:
        hash(d, ignore, _hash=hasher)

    return hasher.hexdigest()


def merge(d1: dict[str, Any], d2: dict[str, Any], _path: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = d1.copy()
    _path = _path or []

    for k, v in d2.items():
        if k not in out or out[k] is None:
            out[k] = v

        elif isinstance(v, dict):
            assert isinstance(d1[k], dict), f"Key type mismatch at {'.'.join(_path)}. Expected dict."
            out[k] = merge(d1[k], d2[k], _path + [k])

        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            out[k] = _zip_longest(merge, d1[k], v)

        else:
            out[k] = v

    return out


def filter(pred: Callable[[Any], bool], d: dict[str, Any]) -> dict[str, Any]:
    out = {}

    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = filter(pred, v)

        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            out[k] = [
                filter(pred, sub)
                for sub in v
            ]

        else:
            if pred(v):
                out[k] = v

    return out


def get_at_path[T](d: dict[str, T], path: str) -> T:
    if '.' not in path:
        return d[path]

    parts = path.split('.')
    sub: Any = d
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            return sub[part]

        if part not in sub:
            raise Exception(f'Item not found at path: {path}')

        sub = sub[part]

    raise Exception(f'Item not found at path: {path}')


def has_path(d: dict[str, Any], path: str) -> bool:
    if '.' not in path:
        return path in d

    parts = path.split('.')
    sub: Any = d
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            return part in sub

        if part not in sub:
            return False

        sub = sub[part]

    return False


def set_at_path[T](d: dict[str, T], path: str, val: T) -> dict[str, T]:
    if '.' not in path:
        d[path] = val
        return d

    parts = path.split('.')

    sub: Any = d
    for i in range(len(parts)):
        part = parts[i]
        is_last = i == len(parts) - 1

        # check if this path component is a list index
        # e.g. b[0] or hi[1] ...
        ls_part = re.match(r'(.+)\[(\d+)\]', part)

        # if last part, then do mutable assignment
        # otherwise keep walking, building subdicts as needed
        if is_last and part not in sub:
            sub[part] = val
        elif is_last and part in sub:
            # if there is already a value
            # ensure the override has a matching type
            t = type(sub[part])
            sub[part] = t(val)
        # otherwise, keep walking and building subdicts
        elif part not in sub and not ls_part:
            sub[part] = {}
            sub = sub[part]
        elif ls_part is not None and ls_part.group(1) not in sub:
            key = ls_part.group(1)
            idx = int(ls_part.group(2))
            sub[key] = [{} for _ in range(idx + 1)]
            sub = sub[key][idx]
        elif ls_part is not None:
            key = ls_part.group(1)
            idx = int(ls_part.group(2))
            sub = sub[key][idx]
        else:
            sub = sub[part]

    return d


def dataclass_to_dict(o: Any) -> Any:
    # For dataclass Thing and instance t=Thing()
    # is_dataclass(Thing) *and* is_dataclass(t)
    # both return True. Therefore
    #   (not isclass(o) and is_dataclass(o))
    # lets us distinguish between the type and the instance
    if not isclass(o) and is_dataclass(o):
        return {
            k: dataclass_to_dict(v)
            for k, v in o.__dict__.items()
        }

    # In the case that an instance instead of a class
    # is given, just return back the underlying values
    if isinstance(o, list):
        return [
            dataclass_to_dict(sub) for sub in o
        ]

    if not is_dataclass(o):
        return o

    # In the case a class is given, we need to reason
    # about its default values and factories.
    out = {}
    for v in fields(o):
        if isinstance(v.default, FieldInfo) and v.default.default_factory is not None:
            factory: Any = v.default.default_factory
            out[v.name] = dataclass_to_dict(factory())

        elif not isinstance(v.default_factory, _MISSING_TYPE):
            out[v.name] = dataclass_to_dict(v.default_factory())

        elif isinstance(v.default, FieldInfo):
            out[v.name] = dataclass_to_dict(v.default.default)

        elif not isinstance(v.default, _MISSING_TYPE):
            out[v.name] = dataclass_to_dict(v.default)

    return out


# ------------------------
# -- Internal Utilities --
# ------------------------
def _zip_longest[T](
    f: Callable[[T, T], T],
    l1: Sequence[T],
    l2: Sequence[T],
) -> list[T]:

    out: list[T] = []
    for i in range(max(len(l1), len(l2))):
        if i >= len(l1):
            out.append(l2[i])
        elif i >= len(l2):
            out.append(l1[i])
        else:
            out.append(f(l1[i], l2[i]))

    return out
