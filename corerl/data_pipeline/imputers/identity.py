import pandas as pd

from dataclasses import dataclass

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group



@dataclass
class IdentityImputerConfig(BaseImputerConfig):
    name: str = "identity"


class IdentityImputer(BaseImputer):
    def __init__(self, cfg: IdentityImputerConfig):
        super().__init__(cfg)

    def _get_imputed_val(self, data: pd.Series | pd.DataFrame, impute_ind: pd.Timestamp) -> float:
        return data.loc[impute_ind]

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


imputer_group.dispatcher(IdentityImputer)
