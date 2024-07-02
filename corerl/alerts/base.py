from omegaconf import DictConfig
from abc import ABC, abstractmethod

class BaseAlert(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, cumulant_start_ind: int, **kwargs):
        self.cumulant_start_ind = cumulant_start_ind

    @abstractmethod
    def evaluate(self, **kwargs) -> dict:
        """
        Feed the latest state/observation/data pertinent to the given alert and evaluate if an alert must be sent to the operator
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """
        Update the alert's value function(s)
        """
        raise NotImplementedError

    @abstractmethod
    def get_dim(self) -> int:
        """
        Returns the number of parallel alerts tracked by the given alert type (Ex: GVF alert tracks multiple endogenous variables)
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