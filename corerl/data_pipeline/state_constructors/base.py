from abc import ABC, abstractmethod

from corerl.data_pipeline.datatypes import Transition


class BaseStateConstructor(ABC):
    @abstractmethod
    def __call__(self, transitions: list[Transition]) -> list[Transition]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

