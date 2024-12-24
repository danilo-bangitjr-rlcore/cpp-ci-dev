from typing import Literal
from pydantic import Field
from corerl.configs.config import config, MISSING, list_
from corerl.data_pipeline.imputers.factory import ImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.null import NullConfig


@config()
class TagConfig:
    name: str = MISSING

    bounds: tuple[float | None, float | None] = (None, None)
    outlier: OddityFilterConfig = Field(default_factory=EMAFilterConfig, discriminator='name')
    imputer: ImputerConfig = Field(default_factory=IdentityImputerConfig, discriminator='name')
    reward_constructor: list[TransformConfig] = list_([NullConfig()])
    state_constructor: list[TransformConfig] | None = None
    tag_type: Literal['action', 'observation', 'meta'] = 'observation'
