from corerl.component.buffer.buffers import (
    EnsembleUniformReplayBufferConfig,
    PriorityReplayBufferConfig,
    buffer_group,
)
from corerl.component.buffer.uniform import UniformReplayBufferConfig

BufferConfig = (
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
)


def init_buffer(cfg: BufferConfig):
    return buffer_group.dispatch(cfg)
