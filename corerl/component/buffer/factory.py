from corerl.component.buffer.buffers import buffer_group, UniformReplayBufferConfig, UniformBuffer # noqa: F401
from corerl.component.buffer.buffers import PriorityBuffer, EnsembleUniformBuffer # noqa: F401


def init_buffer(cfg: UniformReplayBufferConfig) -> UniformBuffer:
    return buffer_group.dispatch(cfg)
