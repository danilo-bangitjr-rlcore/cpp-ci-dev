from src.agent.simple_ac import SimpleAC
from src.agent.sac import SAC
from src.agent.greedy_ac import GreedyAC
from src.agent.reinforce import Reinforce
from src.agent.explore_then_commit import ExploreThenCommit
from src.agent.reseau_exploration_agent import ReseauExplorationAgent
from src.agent.sarsa import Sarsa
from src.agent.line_search import LineSearchGAC, LineSearchReset
from src.agent.inac import InAC
from src.agent.iql import IQL


def init_agent(name, cfg):
    if name == "SimpleAC":
        return SimpleAC(cfg)
    elif name == "SAC":
        return SAC(cfg)
    elif name == "GAC":
        return GreedyAC(cfg)
    elif name == "GAC-OE":
        return GreedyAC(cfg, average_entropy=False)
    elif name == "Reinforce":
        return Reinforce(cfg)
    elif name == "ETC":
        return ExploreThenCommit(cfg)
    elif name == "Reseau-Exploration":
        return ReseauExplorationAgent(cfg)
    elif name == "sarsa":
        return Sarsa(cfg)
    elif name == "LineSearchGAC":
        return LineSearchGAC(cfg)
    elif name == "LineSearchReset":
        return LineSearchReset(cfg)

    elif name == "IQL":
        return IQL(cfg)
    elif name == "InAC":
        return InAC(cfg)
    else:
        raise NotImplementedError