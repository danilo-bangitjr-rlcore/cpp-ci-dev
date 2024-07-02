import json
from pathlib import Path

from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseEval(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def do_eval(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
        raise NotImplementedError

    def output(self, path: Path) -> None:
        stats = self.get_stats()

        print(stats)

        with open(path, 'w') as f:
            json.dump(stats, f)
