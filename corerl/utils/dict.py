from collections.abc import MutableMapping, Sequence


def drop(d: MutableMapping, to_drop: Sequence[str]) -> dict:
    to_keep = set(d.keys()) - set(to_drop)
    return {
        k: d[k] for k in to_keep
     }


def flatten(d: MutableMapping, parent_key: str = "", separator: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)
