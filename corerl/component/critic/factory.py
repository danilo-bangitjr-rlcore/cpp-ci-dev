from corerl.component.critic.base_critic import (
    CriticConfig,
    EnsembleCritic,
)
from corerl.state import AppState


def init_q_critic(
        cfg: CriticConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
        output_dim: int = 1,
    ) -> EnsembleCritic:
    return EnsembleCritic(cfg, app_state, state_dim, action_dim, output_dim)
