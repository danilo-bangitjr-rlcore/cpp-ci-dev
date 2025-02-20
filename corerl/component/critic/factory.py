from corerl.component.critic.ensemble_critic import (
    EnsembleCriticConfig,
    EnsembleQCritic,
    EnsembleVCritic,
)
from corerl.state import AppState


def init_q_critic(
        cfg: EnsembleCriticConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
        output_dim: int = 1,
    ) -> EnsembleQCritic:
    """
    corresponding configs: config/agent/critic
    """
    if cfg.name == 'ensemble':
        critic = EnsembleQCritic(cfg, app_state, state_dim, action_dim, output_dim)
    else:
        raise NotImplementedError

    return critic


def init_v_critic(
    cfg: EnsembleCriticConfig,
    app_state: AppState,
    state_dim: int,
    output_dim: int = 1
    ) -> EnsembleVCritic:
    """
    corresponding configs: config/agent/critic
    """
    if cfg.name == 'ensemble':
        critic = EnsembleVCritic(cfg, app_state, state_dim, output_dim)
    else:
        raise NotImplementedError

    return critic
