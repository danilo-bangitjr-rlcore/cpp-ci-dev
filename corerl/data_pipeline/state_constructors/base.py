from dataclasses import dataclass
from abc import ABC, abstractmethod

from corerl.data.data import Transition


class BaseStateConstructor(ABC):
    @abstractmethod
    def __call__(self, transitions: list[Transition]) -> list[Transition]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

