from typing import Annotated

from pydantic import Field

from corerl.data_pipeline.imputers.auto_encoder import MaskedAEConfig, MaskedAutoencoder
from corerl.data_pipeline.imputers.base import imputer_group
from corerl.data_pipeline.imputers.imputer_stage import PerTagImputer, PerTagImputerConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.state import AppState

imputer_group.dispatcher(PerTagImputer)
imputer_group.dispatcher(MaskedAutoencoder)

ImputerStageConfig = Annotated[(
    PerTagImputerConfig
    | MaskedAEConfig
), Field(discriminator='name')]

def init_imputer(imputer_cfg: ImputerStageConfig, app_state: AppState, tag_cfgs: list[TagConfig]):
    return imputer_group.dispatch(imputer_cfg, app_state, tag_cfgs)
