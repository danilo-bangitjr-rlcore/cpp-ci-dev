from lib_config.config import config
from pydantic import Field

from corerl.data_pipeline.datatypes import DataMode, StageCode


@config()
class RawDataEvalConfig:
    data_modes: list[DataMode] = Field(default_factory=lambda:[DataMode.ONLINE])
    stage_codes: list[StageCode] = Field(default_factory=lambda:[StageCode.INIT])
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
