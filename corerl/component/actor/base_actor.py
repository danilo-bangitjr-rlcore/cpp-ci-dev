import torch
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseActor(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        state: torch.Tensor,
        with_grad: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError
