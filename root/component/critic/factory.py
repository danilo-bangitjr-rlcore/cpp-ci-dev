from ensemble_critic import EnsembleQCritic, EnsembleVCritic


def init_q_critic(cfg, state_dim, action_dim):
    if cfg.name == 'ensemble':
        critic = EnsembleQCritic(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return critic


def init_v_critic(cfg, state_dim):
    if cfg.name == 'ensemble':
        critic = EnsembleVCritic(cfg, state_dim)
    else:
        raise NotImplementedError

    return critic
