import json
from pathlib import Path

from abc import ABC, abstractmethod

class EvalConfig:
    offline_eval: bool
    online_eval: bool


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
