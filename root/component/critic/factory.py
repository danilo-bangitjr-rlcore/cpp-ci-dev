from root.component.critic.ensemble_critic import EnsembleQCritic, EnsembleVCritic
from root.component.critic.ensemble_critic import BaseQ, BaseV
from omegaconf import DictConfig


def init_q_critic(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseQ:
    if cfg.name == 'ensemble':
        critic = EnsembleQCritic(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return critic


def init_v_critic(cfg: DictConfig, state_dim: int) -> BaseV:
    if cfg.name == 'ensemble':
        critic = EnsembleVCritic(cfg, state_dim)
    else:
        raise NotImplementedError

    return critic
