from typing import Annotated

from pydantic import Field

from corerl.configs.data_pipeline.imputers.auto_encoder import MaskedAEConfig
from corerl.configs.data_pipeline.imputers.imputer_stage import PerTagImputerConfig

ImputerStageConfig = Annotated[
    MaskedAEConfig | PerTagImputerConfig,
    Field(discriminator="name"),
]
