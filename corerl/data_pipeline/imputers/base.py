import pandas as pd
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group

from corerl.data_pipeline.datatypes import PipelineFrame


@dataclass
class BaseImputerConfig:
    name: str = MISSING


class BaseImputer(ABC):
    def __init__(self, cfg: BaseImputerConfig):
        self.cfg = cfg

    @abstractmethod
    def _get_imputed_val(self, data: pd.Series | pd.DataFrame, ind: pd.Timestamp) -> float:
        raise NotImplementedError

    def _get_imputed_vals(self, data: pd.Series | pd.DataFrame, imputed_inds: pd.DatetimeIndex) -> np.ndarray:
        imputed_vals = []
        for ind in imputed_inds:
            imputed_val = self._get_imputed_val(data, ind)
            imputed_vals.append(imputed_val)

        return np.array(imputed_vals)

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        raise NotImplementedError


imputer_group = Group[
    [], BaseImputer
]('pipeline/imputer')
