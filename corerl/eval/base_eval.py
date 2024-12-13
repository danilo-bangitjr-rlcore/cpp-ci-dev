import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

@dataclass
class EvalConfig:
    offline_eval: bool = MISSING
    online_eval: bool = MISSING


class BaseEval(ABC):
    @abstractmethod
    def do_eval(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
        raise NotImplementedError

    def output(self, path: Path) -> None:
        stats = self.get_stats()
        with open(path, 'w') as f:
            json.dump(stats, f)
