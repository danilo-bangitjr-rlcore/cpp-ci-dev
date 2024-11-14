from corerl.component.exploration.random_network import explore_group
from corerl.utils.hydra import DiscriminatedUnion


def init_exploration_module(cfg: DiscriminatedUnion, state_dim: int, action_dim: int):
    return explore_group.dispatch(cfg, state_dim, action_dim)
