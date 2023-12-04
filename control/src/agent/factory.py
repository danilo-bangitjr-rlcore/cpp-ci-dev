from src.agent.simple_ac import SimpleAC
from src.agent.sac import SAC
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.agent.reinforce import Reinforce


def init_agent(name, cfg):
    if name == "SimpleAC":
        return SimpleAC(cfg)
    elif name == "SAC":
        return SAC(cfg)
    elif name == "GAC":
        if cfg.discrete_control:
            return GreedyACDiscrete(cfg)
        else:
            return GreedyAC(cfg)
    elif name == "Reinforce":
        return Reinforce(cfg)
    else:
        raise NotImplementedError