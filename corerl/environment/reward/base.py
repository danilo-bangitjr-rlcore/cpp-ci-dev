import pandas as pd
from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):
    """
    Reward class to be used in dataloader and real world environments.
    Want each reward function's __call__() method to have same signature so that they can be
    used interchangeably in the dataloader.
    """
    @abstractmethod
    def __call__(self, obs: pd.DataFrame | pd.Series, **kwargs: Any) -> float:
        raise NotImplementedError
