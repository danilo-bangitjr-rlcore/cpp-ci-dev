from src.agent.simple_ac import SimpleAC
from src.agent.sac import SAC
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.agent.reinforce import Reinforce
from src.agent.greedy_ac_wm import GACPredictSuccess, GACwHardMemory
from src.agent.gac_inac import GAC_InAC


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
    elif name == "GAC-OE":
        if cfg.discrete_control:
            return GreedyACDiscrete(cfg, average_entropy=False)
        else:
            return GreedyAC(cfg, average_entropy=False)
    elif name == "Reinforce":
        return Reinforce(cfg)
    elif name == "GACMH":
        if cfg.discrete_control:
            raise NotImplementedError
        else:
            return GACwHardMemory(cfg, average_entropy=True)
    elif name == "GACPS":
        return GACPredictSuccess(cfg, average_entropy=True)
    elif name == "GACIn":
        return GAC_InAC(cfg, average_entropy=True)
    else:
        raise NotImplementedError