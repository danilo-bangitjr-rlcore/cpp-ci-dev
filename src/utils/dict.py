from typing import Any


def flatten(d: dict[str, object], path: str = '', _out: dict[str, Any] | None = None) -> dict[str, Any]:
    out = _out if _out is not None else {}

    for k, v in d.items():
        if path == '':
            key = k
        else:
            key = path + '.' + k

        if isinstance(v, dict):
            flatten(v, key, out)
        else:
            out[key] = v

    return out
