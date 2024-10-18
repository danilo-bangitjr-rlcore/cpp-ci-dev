from corerl.component.critic.ensemble_critic import EnsembleQCritic, EnsembleVCritic, EnsembleQCriticLineSearch
from omegaconf import DictConfig


def init_q_critic(cfg: DictConfig, state_dim: int, action_dim: int, output_dim: int = 1) -> EnsembleQCritic:
    """
    corresponding configs: config/agent/critic
    """
    if cfg.name == 'ensemble':
        critic = EnsembleQCritic(cfg, state_dim, action_dim, output_dim)
    elif cfg.name == 'ensemble_linesearch':
        critic = EnsembleQCriticLineSearch(cfg, state_dim, action_dim, output_dim)
    else:
        raise NotImplementedError

    return critic


def init_v_critic(cfg: DictConfig, state_dim: int, output_dim: int = 1) -> EnsembleVCritic:
    """
    corresponding configs: config/agent/critic
    """
    if cfg.name == 'ensemble':
        critic = EnsembleVCritic(cfg, state_dim, output_dim)
    else:
        raise NotImplementedError

    return critic
