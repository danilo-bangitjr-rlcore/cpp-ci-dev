
from lib_config.config import config
from pydantic import Field

from corerl.configs.data_pipeline.transforms.delta import DeltaConfig


@config()
class DeltaStageConfig:
    delta_cfg: DeltaConfig = Field(default_factory=DeltaConfig)
