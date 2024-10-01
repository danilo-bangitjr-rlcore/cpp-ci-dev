from __future__ import annotations

import hashlib
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any

import corerl.utils.nullable as nullable


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
