from corerl.component.buffer.buffers import (
    EnsembleUniformReplayBufferConfig,
    PriorityReplayBufferConfig,
    buffer_group,
    UniformReplayBufferConfig,
    UniformBuffer,
)


BufferConfig = (
    UniformReplayBufferConfig
    | PriorityReplayBufferConfig
    | EnsembleUniformReplayBufferConfig
)


def init_buffer(cfg: BufferConfig) -> UniformBuffer:
    return buffer_group.dispatch(cfg)
