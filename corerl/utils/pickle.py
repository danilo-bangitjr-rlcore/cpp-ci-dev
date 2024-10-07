import pickle
from pathlib import Path


def maybe_load(path: Path) -> object | None:
    if not path.exists():
        return

    with open(path, 'rb') as f:
        return pickle.load(f)


def dump(path: Path, obj: object):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(obj, f)
