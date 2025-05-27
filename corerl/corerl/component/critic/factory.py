from corerl.component.critic.gtd_critic import GTDCritic, GTDCriticConfig
from corerl.state import AppState


def create_critic(
    cfg: GTDCriticConfig,
    app_state: AppState,
    state_dim: int,
    action_dim: int,
):
    return GTDCritic(cfg, app_state, state_dim, action_dim)
