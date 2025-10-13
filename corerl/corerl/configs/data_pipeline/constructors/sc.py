
from __future__ import annotations

from lib_config.config import config, list_
from pydantic import Field

from corerl.configs.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.configs.data_pipeline.transforms import TraceConfig, TransformConfig


@config()
class SCConfig:
    defaults: list[TransformConfig] = list_([TraceConfig()])
    countdown: CountdownConfig | None = Field(default_factory=CountdownConfig)
