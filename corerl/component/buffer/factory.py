from corerl.component.buffer.buffers import (
    EnsembleUniformReplayBufferConfig,
    PriorityReplayBufferConfig,
    UniformReplayBufferConfig,
    buffer_group,
)

BufferConfig = (
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
)


def init_buffer(cfg: BufferConfig):
    return buffer_group.dispatch(cfg)
