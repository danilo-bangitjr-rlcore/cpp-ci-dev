from typing import Annotated

from pydantic import Field

from corerl.component.buffer.ensemble import (
    EnsembleUniformReplayBufferConfig,
    buffer_group,
)
from corerl.component.buffer.mixed_history import MixedHistoryBufferConfig
from corerl.component.buffer.priority import PriorityReplayBufferConfig
from corerl.component.buffer.uniform import UniformReplayBufferConfig
from corerl.state import AppState

BufferConfig = Annotated[
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
    | MixedHistoryBufferConfig
, Field(discriminator='name')]


def init_buffer(cfg: BufferConfig, app_state: AppState):
    return buffer_group.dispatch(cfg, app_state)
