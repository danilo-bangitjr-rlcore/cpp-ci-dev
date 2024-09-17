from collections.abc import MutableMapping, Sequence


def drop(d: MutableMapping, to_drop: Sequence[str]) -> dict:
    to_keep = set(d.keys()) - set(to_drop)
    return {
        k: d[k] for k in to_keep
     }


def flatten(dictionary: MutableMapping, parent_key: str = "", separator: str = "_") -> dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
