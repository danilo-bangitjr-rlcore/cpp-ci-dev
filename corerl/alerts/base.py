from typing import Any
from omegaconf import DictConfig
from abc import ABC, abstractmethod

import torch

from corerl.data.data import Transition

class BaseAlert(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        self.cumulant_start_ind = cumulant_start_ind

    @abstractmethod
    def evaluate(self, **kwargs) -> dict:
        """
        Feed the latest state/observation/data pertinent to the given alert
        and evaluate if an alert must be sent to the operator
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> dict:
        """
        Update the alert's value function(s)
        """
        raise NotImplementedError

    @abstractmethod
    def get_dim(self) -> int:
        """
        Returns the number of parallel alerts tracked by the given alert type
        (Ex: GVF alert tracks multiple endogenous variables)
        """
        raise NotImplementedError

    @abstractmethod
    def get_discount_factors(self) -> list[float]:
        """
        Returns the list of discount factors used in the given alert type
        """
        raise NotImplementedError

    @abstractmethod
    def get_cumulants(self, **kwargs) -> list[float]:
        """
        Utilizes the info in **kwargs to produce the list of cumulants for the given alert type
        """
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition: Transition):
        ...

    @abstractmethod
    def load_buffer(self, transitions: list[Transition]):
        ...

    @abstractmethod
    def get_trace_thresh(self) -> dict[str, dict[str, float]]:
        ...

    @abstractmethod
    def get_std_thresh(self) -> dict[str, dict[str, float]]:
        ...

    @abstractmethod
    def alert_type(self) -> str:
        ...

    @abstractmethod
    def get_test_state_qs(
        self,
        plot_info: dict[str, Any],
        repeated_test_states: torch.Tensor,
        repated_actions: torch.Tensor,
        num_states: int,
        test_actions: int,
    ) -> dict[str, Any]: ...


    @abstractmethod
    def get_buffer_size(self) -> list[int]:
        ...
