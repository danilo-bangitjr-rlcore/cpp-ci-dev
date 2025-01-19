from pydantic import Field
from typing_extensions import Annotated

from corerl.data_pipeline.imputers.per_tag.base import BasePerTagImputer, BasePerTagImputerConfig, per_tag_imputer_group
from corerl.data_pipeline.imputers.per_tag.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.per_tag.identity import IdentityImputerConfig
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputerConfig

ImputerConfig = Annotated[
    IdentityImputerConfig
    | CopyImputerConfig
    | LinearImputerConfig
, Field(discriminator='name')]


def init_per_tag_imputer(cfg: BasePerTagImputerConfig) -> BasePerTagImputer:
    return per_tag_imputer_group.dispatch(cfg)
