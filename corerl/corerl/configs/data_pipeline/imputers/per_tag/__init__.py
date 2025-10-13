from typing import Annotated

from pydantic import Field

from corerl.configs.data_pipeline.imputers.per_tag.backfill import BackfillImputerConfig
from corerl.configs.data_pipeline.imputers.per_tag.copy import CopyImputerConfig
from corerl.configs.data_pipeline.imputers.per_tag.identity import IdentityImputerConfig
from corerl.configs.data_pipeline.imputers.per_tag.linear import LinearImputerConfig

ImputerConfig = Annotated[
    IdentityImputerConfig | CopyImputerConfig | LinearImputerConfig | BackfillImputerConfig,
    Field(discriminator='name'),
]
