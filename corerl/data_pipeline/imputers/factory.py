from typing import Annotated

from pydantic import Field

from corerl.data_pipeline.imputers.base import imputer_group
from corerl.data_pipeline.imputers.imputer_stage import PerTagImputer, PerTagImputerConfig
from corerl.data_pipeline.tag_config import TagConfig

imputer_group.dispatcher(PerTagImputer)

ImputerStageConfig = Annotated[(
    PerTagImputerConfig
), Field(discriminator='name')]

def init_imputer(imputer_cfg: ImputerStageConfig, tag_cfgs: list[TagConfig]):
    return imputer_group.dispatch(imputer_cfg, tag_cfgs)
