from corerl.component.buffer.ensemble import (
    EnsembleUniformReplayBufferConfig,
    buffer_group,
)
from corerl.component.buffer.priority import PriorityReplayBufferConfig
from corerl.component.buffer.uniform import UniformReplayBufferConfig

BufferConfig = (
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
)


def init_buffer(cfg: BufferConfig):
    return buffer_group.dispatch(cfg)
