from corerl.component.buffer.buffers import (
    EnsembleUniformReplayBufferConfig,
    PriorityReplayBufferConfig,
    UniformBuffer,
    UniformReplayBufferConfig,
    buffer_group,
)

BufferConfig = (
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
)


def init_buffer(cfg: BufferConfig) -> UniformBuffer:
    return buffer_group.dispatch(cfg)
