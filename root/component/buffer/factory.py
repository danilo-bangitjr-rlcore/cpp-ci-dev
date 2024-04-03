import root.component.buffer.buffers as buffers
from omegaconf import DictConfig


def init_buffer(cfg: DictConfig) -> buffers.UniformBuffer:
    name = cfg.name
    if name == 'Prioritized':
        buffer = buffers.PriorityBuffer(cfg)
    elif name == 'Uniform':
        buffer = buffers.UniformBuffer(cfg)
    else:
        raise NotImplementedError

    return buffer
