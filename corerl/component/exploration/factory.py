from corerl.component.exploration.random_network import RndNetworkExploreConfig, explore_group


ExploreModuleConfig = RndNetworkExploreConfig

def init_exploration_module(cfg: ExploreModuleConfig, state_dim: int, action_dim: int):
    return explore_group.dispatch(cfg, state_dim, action_dim)
