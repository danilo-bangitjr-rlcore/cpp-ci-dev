from corerl.component.actor.base_actor import BaseActor, group
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.state import AppState

ActorConfig = NetworkActorConfig

def init_actor(
    cfg: ActorConfig,
    app_state : AppState,
    state_dim: int,
    action_dim: int,
    initializer: BaseActor | None = None,
):
    return group.dispatch(cfg, app_state, state_dim, action_dim, initializer)
