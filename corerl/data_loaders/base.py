import numpy as np
import pickle as pkl
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import pandas as pd
from corerl.data.data import ObsTransition, OldObsTransition
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.environment.reward.base import BaseReward
from corerl.utils.hydra import Group, list_


@dataclass
class BaseDataLoaderConfig:
    name: str = 'base'
    offline_data_path: str = 'offline_data'
    train_filenames: list[str] = list_()
    test_filenames: list[str] = list_()


class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, cfg: BaseDataLoaderConfig, _):
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
    def load_data(
        self,
        filenames: Sequence[str] | Sequence[Path],
    ) -> pd.DataFrame:
        """
        Read offline data into a single dataframe sorted by date, containing only columns in observation space
        """
        raise NotImplementedError

    @abstractmethod
    def create_obs_transitions(
        self,
        df: pd.DataFrame,
        reward_function: BaseReward, *args,
    ) -> list[OldObsTransition] | list[ObsTransition]:
        raise NotImplementedError

    def save(self, save_obj: object, path: Path):
        with open(path, "wb") as file:
            pkl.dump(save_obj, file)

    def load(self, path: Path) -> object:
        with open(path, "rb") as file:
            obj = pkl.load(file)
            return obj

    @abstractmethod
    def get_obs_max_min(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class OldBaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, cfg: BaseDataLoaderConfig, _):
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
    def load_data(
        self,
        filenames: Sequence[str] | Sequence[Path],
    ) -> pd.DataFrame:
        """
        Read offline data into a single dataframe sorted by date, containing only columns in observation space
        """
        raise NotImplementedError

    @abstractmethod
    def create_obs_transitions(
        self,
        df: pd.DataFrame,
        normalizer: ObsTransitionNormalizer,
        reward_function: BaseReward, *args,
    ) -> list[OldObsTransition]:
        raise NotImplementedError

    def save(self, save_obj: object, path: Path):
        with open(path, "wb") as file:
            pkl.dump(save_obj, file)

    def load(self, path: Path) -> object:
        with open(path, "rb") as file:
            obj = pkl.load(file)
            return obj

    @abstractmethod
    def get_obs_max_min(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# set up config groups
dl_group = Group[[], OldBaseDataLoader | BaseDataLoader](
    'data_loader',
)
