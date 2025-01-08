from typing_extensions import Annotated
from pydantic import Field

from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
from corerl.data_pipeline.imputers.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.imputers.linear import LinearImputerConfig

ImputerConfig = Annotated[
    IdentityImputerConfig
    | CopyImputerConfig
    | LinearImputerConfig
, Field(discriminator='name')]


def init_imputer(cfg: BaseImputerConfig) -> BaseImputer:
    return imputer_group.dispatch(cfg)
