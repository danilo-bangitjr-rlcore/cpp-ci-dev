from omegaconf import DictConfig
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from corerl.environment.reward.base import BaseReward
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer_utils import BaseNormalizer

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
    def create_transitions(self, df: pd.DataFrame, state_constructor: BaseStateConstructor, reward_function: BaseReward, action_normalizer: BaseNormalizer, reward_normalizer: BaseNormalizer) -> list[tuple]:
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self, transitions: list[tuple], shuffle: bool=True) -> (list[tuple], list[tuple]):
        raise NotImplementedError

    @abstractmethod
    def save_transitions(self, transitions: list[tuple], path: Path):
        raise NotImplementedError

    @abstractmethod
    def load_transitions(self, path: Path) -> list[tuple]:
        raise NotImplementedError