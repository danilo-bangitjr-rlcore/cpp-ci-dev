from ml_instrumentation.Collector import Collector

from agent.gac import GreedyAC, GreedyACConfig


def get_agent(cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
    """Creates an agent instance based on configuration name."""
    if cfg.name.lower() == "gac":
        return GreedyAC(cfg, seed, state_dim, action_dim, collector)

    raise ValueError(f"Unknown agent type: {cfg.name}. Supported agents: 'gac', 'gaac'")
