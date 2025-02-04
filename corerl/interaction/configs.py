from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field

from corerl.configs.config import MISSING, config
from corerl.messages.heartbeat import HeartbeatConfig


# -------------
# -- Configs --
# -------------
@config()
class BaseInteractionConfig:
    obs_period: timedelta = MISSING
    action_period: timedelta = MISSING
    action_tolerance: timedelta = MISSING


@config()
class SimInteractionConfig(BaseInteractionConfig):
    name: Literal["sim_interaction"] = "sim_interaction"


@config()
class DepInteractionConfig(BaseInteractionConfig):
    name: Literal["dep_interaction"] = "dep_interaction"
    historical_batch_size: int = 10000
    hist_chunk_start: datetime | None = None
    checkpoint_path: Path = Path('outputs/checkpoints')
    restore_checkpoint: bool = True
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    warmup_period: timedelta | None = None
