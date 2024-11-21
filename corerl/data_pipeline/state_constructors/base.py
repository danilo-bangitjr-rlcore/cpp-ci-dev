from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group

from corerl.data_pipeline.datatypes import Transition


@dataclass
class BaseStateConstructorConfig:
    name: str = MISSING


class BaseStateConstructor(ABC):
    def __init__(self, cfg: BaseStateConstructorConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, transitions: list[Transition], tag: str) -> list[Transition]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


state_constructor_group = Group[
    [], BaseStateConstructor
]('pipeline/state_constructor')
