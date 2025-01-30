from typing import Annotated

from pydantic import Field

from corerl.component.actor.base_actor import BaseActor, group
from corerl.component.actor.network_actor import NetworkActorConfig

ActorConfig = NetworkActorConfig

def init_actor(
    cfg: ActorConfig,
    state_dim: int,
    action_dim: int,
    initializer: BaseActor | None = None,
):
    return group.dispatch(cfg, state_dim, action_dim, initializer)
