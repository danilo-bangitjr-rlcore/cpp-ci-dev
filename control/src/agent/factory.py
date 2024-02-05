from src.agent.simple_ac import SimpleAC
from src.agent.sac import SAC
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.agent.reinforce import Reinforce
from src.agent.explore_then_commit import ExploreThenCommit
from src.agent.reseau_exploration_agent import ReseauExplorationAgent
from src.agent.sarsa import Sarsa


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
    elif name == "ETC":
        return ExploreThenCommit(cfg)
    elif name == "Reseau-Exploration":
        return ReseauExplorationAgent(cfg)
    elif name == "sarsa":
        return Sarsa(cfg)
    else:
        raise NotImplementedError