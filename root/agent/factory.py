from root.agent.simple_ac import SimpleAC


def init_agent(cfg, state_dim, action_dim, seed):
    if cfg.name == 'SimpleAC':
        agent = SimpleAC(cfg, state_dim, action_dim, seed)
    else:
        raise NotImplementedError

    return agent
