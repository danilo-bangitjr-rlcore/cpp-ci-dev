from corerl.utils.hydra import DiscriminatedUnion
from corerl.component.actor.base_actor import group
from corerl.component.actor.network_actor import NetworkActor


def init_actor(
    cfg: DiscriminatedUnion,
    state_dim: int,
    action_dim: int,
    initializer: NetworkActor | None = None,
):
    return group.dispatch(cfg, state_dim, action_dim, initializer)
