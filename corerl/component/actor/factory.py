from corerl.utils.hydra import DiscriminatedUnion
from corerl.component.actor.base_actor import group, BaseActor

import corerl.component.actor.network_actor # noqa: F401

def init_actor(
    cfg: DiscriminatedUnion,
    state_dim: int,
    action_dim: int,
    initializer: BaseActor | None = None,
):
    return group.dispatch(cfg, state_dim, action_dim, initializer)
