from corerl.component.critic.ensemble_critic import EnsembleCritic, SARSACriticConfig
from corerl.component.critic.gtd_critic import GTDCritic, GTDCriticConfig
from corerl.state import AppState


def create_critic(
    cfg: GTDCriticConfig | SARSACriticConfig,
    app_state: AppState,
    state_dim: int,
    action_dim: int,
):
    if isinstance(cfg, GTDCriticConfig):
        return GTDCritic(cfg, app_state, state_dim, action_dim)
    else:
        return EnsembleCritic(cfg, app_state, state_dim, action_dim)
