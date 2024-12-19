from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
from corerl.data_pipeline.imputers.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.imputers.linear import LinearImputerConfig

ImputerConfig = (
    IdentityImputerConfig
    | CopyImputerConfig
    | LinearImputerConfig
)


def init_imputer(cfg: BaseImputerConfig) -> BaseImputer:
    return imputer_group.dispatch(cfg)
