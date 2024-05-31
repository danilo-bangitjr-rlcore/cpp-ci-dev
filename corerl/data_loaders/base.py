import numpy as np
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from corerl.environment.reward.base import BaseReward
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.base import BaseInteraction


class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Read offline data into a single dataframe sorted by date, containing only columns in observation space
        """
        raise NotImplementedError

    @abstractmethod
    def create_transitions(self,
                           df: pd.DataFrame,
                           state_constructor: BaseStateConstructor,
                           reward_function: BaseReward,
                           interaction: BaseInteraction) -> dict:
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self, transitions: list[tuple], shuffle: bool = True) -> (list[tuple], list[tuple]):
        raise NotImplementedError

    @abstractmethod
    def save(self, lst: object, path: Path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> object:
        raise NotImplementedError

    @abstractmethod
    def get_obs_max_min(self, offline_data_df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        raise NotImplementedError
