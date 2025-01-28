from typing import Annotated

from pydantic import Field

from corerl.component.buffer.ensemble import (
    EnsembleUniformReplayBufferConfig,
    buffer_group,
)
from corerl.component.buffer.priority import PriorityReplayBufferConfig
from corerl.component.buffer.uniform import UniformReplayBufferConfig

BufferConfig = Annotated[
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
, Field(discriminator='name')]


def init_buffer(cfg: BufferConfig):
    return buffer_group.dispatch(cfg)
