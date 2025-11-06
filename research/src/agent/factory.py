from lib_agent.gamma_schedule import GammaScheduler
from ml_instrumentation.Collector import Collector

from agent.gac import GreedyAC, GreedyACConfig
from src.agent.gaac import GAAC


def get_agent(
    cfg: GreedyACConfig,
    seed: int,
    state_dim: int,
    action_dim: int,
    collector: Collector,
    gamma_scheduler: GammaScheduler,
):
    """Creates an agent instance based on configuration name."""
    if cfg.name.lower() == "gac":
        return GreedyAC(cfg, seed, state_dim, action_dim, collector, gamma_scheduler)
    if cfg.name.lower() == "gaac":
        return GAAC(cfg, seed, state_dim, action_dim, collector, gamma_scheduler)

    raise ValueError(f"Unknown agent type: {cfg.name}. Supported agents: 'gac', 'gaac'")
