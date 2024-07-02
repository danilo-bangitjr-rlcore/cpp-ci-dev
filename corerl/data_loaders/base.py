import numpy as np
import pickle as pkl
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
        self.offline_data_path = Path(cfg.offline_data_path)
        # You can either load all the csvs in the directory or a subset

        if not cfg.train_filenames:
            # will return all files as training data
            self.train_filenames = list(self.offline_data_path.glob('*.csv'))
            self.test_filenames = []
        else:
            self.train_filenames = [self.offline_data_path / file for file in cfg.train_filenames]
            self.test_filenames = [self.offline_data_path / file for file in cfg.test_filenames]

        self.all_filenames = self.train_filenames + self.test_filenames

        raise NotImplementedError

    @abstractmethod
    def load_data(self, filenames: list[str]) -> pd.DataFrame:
        """
        Read offline data into a single dataframe sorted by date, containing only columns in observation space
        """
        raise NotImplementedError

    @abstractmethod
    def create_obs_transitions(self,
                               df: pd.DataFrame,
                               interaction: BaseInteraction,
                               reward_function: BaseReward, *args) -> dict:
        raise NotImplementedError

    def save(self, save_obj: object, path: Path):
        with open(path, "wb") as file:
            pkl.dump(save_obj, file)

    def load(self, path: Path) -> object:
        with open(path, "rb") as file:
            obj = pkl.load(file)
            return obj

    @abstractmethod
    def get_obs_max_min(self, offline_data_df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        raise NotImplementedError
