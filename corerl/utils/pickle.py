import pickle
from pathlib import Path


def maybe_load(path: Path) -> object | None:
    if not path.exists():
        return

    with open(path, 'rb') as f:
        return pickle.load(f)
