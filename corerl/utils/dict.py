from collections.abc import MutableMapping, Sequence


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
