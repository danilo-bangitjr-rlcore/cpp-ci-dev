from collections.abc import Hashable
from pathlib import Path
import pickle as pkl
from typing import Any

freezer = None


class Freezer:
    def __init__(self, save_path: Path):
        super().__init__()
        self.dict = dict[Hashable, Any]()
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def __getitem__(self, key: Hashable):
        return self.dict[key]

    def store(self, key: Hashable, value: Any):
        if key in self.dict:
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]

    def save(self):
        with open(self.save_path / 'step-{}.pkl'.format(self.step), 'wb') as f:
            pkl.dump(self.dict, f)

    def increment(self):
        self.step += 1

    def clear(self):
        self.dict = dict()


def init_freezer(save_path: Path):
    global freezer
    freezer = Freezer(save_path)
