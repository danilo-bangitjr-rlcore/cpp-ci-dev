import buffers

def init_buffer(cfg, seed):
    name = cfg.name
    if name == 'Prioritized':
        buffer = buffers.PriorityBuffer(cfg, seed)
    elif name == 'Uniform':
        buffer = buffers.UniformBuffer(cfg, seed)
    else:
        raise NotImplementedError

    return buffer