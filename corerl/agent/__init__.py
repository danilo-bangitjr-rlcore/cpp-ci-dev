from corerl.agent.action_schedule import ActionScheduleAgent
from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import ExploreLSGAC, GreedyAC, GreedyACLineSearch
from corerl.agent.greedy_iql import GreedyIQL
from corerl.agent.inac import InAC
from corerl.agent.iql import IQL
from corerl.agent.random import RandomAgent
from corerl.agent.reinforce import Reinforce
from corerl.agent.sac import SAC
from corerl.agent.sarsa import EpsilonGreedySarsa
from corerl.agent.simple_ac import SimpleAC
from corerl.utils.hydra import Group

agent_group = Group[[int, int],BaseAgent,]('agent')

agent_group.dispatcher(ActionScheduleAgent)
agent_group.dispatcher(GreedyAC)
agent_group.dispatcher(GreedyACLineSearch)
agent_group.dispatcher(ExploreLSGAC)
agent_group.dispatcher(GreedyIQL)
agent_group.dispatcher(InAC)
agent_group.dispatcher(IQL)
agent_group.dispatcher(RandomAgent)
agent_group.dispatcher(Reinforce)
agent_group.dispatcher(SAC)
agent_group.dispatcher(EpsilonGreedySarsa)
agent_group.dispatcher(SimpleAC)
