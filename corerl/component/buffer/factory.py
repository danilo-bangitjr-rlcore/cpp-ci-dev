import corerl.component.buffer.buffers as buffers
from omegaconf import DictConfig


def init_buffer(cfg: DictConfig) -> buffers.UniformBuffer:
    """
    corresponding configs: part of config/agent
    """
    print("Buffer CFG:")
    print(cfg)
    name = cfg.name
    if name == 'prioritized':
        buffer = buffers.PriorityBuffer(cfg)
    elif name == 'uniform':
        buffer = buffers.UniformBuffer(cfg)
    elif name == 'ensemble_uniform':
        buffer = buffers.EnsembleUniformBuffer(cfg)
    else:
        raise NotImplementedError

    return buffer
