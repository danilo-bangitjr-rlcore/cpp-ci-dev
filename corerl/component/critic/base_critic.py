import torch
from pathlib import Path
from abc import ABC, abstractmethod
from corerl.configs.config import config


@config()
class BaseCriticConfig:
    discrete_control: bool = False


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: BaseCriticConfig):
        self.discrete_control = cfg.discrete_control

    @abstractmethod
    def update(self, loss: list[torch.Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


@config()
class BaseVConfig(BaseCriticConfig):
    ...


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: BaseVConfig, state_dim: int):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(
        self,
        state_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_v_target(
        self,
        state_batches: list[torch.Tensor],
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError


@config()
class BaseQConfig(BaseCriticConfig):
    ...


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: BaseQConfig, state_dim: int, action_dim: int):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_q_target(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError
