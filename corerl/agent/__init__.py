from corerl.agent.base import BaseAgent
from corerl.configs.group import Group
from corerl.state import AppState

from corerl.agent.action_schedule import ActionScheduleAgent, ActionScheduleConfig
from corerl.agent.greedy_ac import GreedyAC, GreedyACConfig
from corerl.agent.greedy_iql import GreedyIQL, GreedyIQLConfig
from corerl.agent.inac import InAC, InACConfig
from corerl.agent.iql import IQL, IQLConfig
from corerl.agent.random import RandomAgent, RandomAgentConfig
from corerl.agent.sac import SAC, SACConfig
from corerl.agent.sarsa import EpsilonGreedySarsa, EpsilonGreedySarsaConfig
from corerl.agent.simple_ac import SimpleAC, SimpleACConfig

agent_group = Group[
    [AppState, int, int],
    BaseAgent,
]()

AgentConfig = (
    ActionScheduleConfig
    | GreedyACConfig
    | GreedyIQLConfig
    | InACConfig
    | IQLConfig
    | RandomAgentConfig
    | SACConfig
    | EpsilonGreedySarsaConfig
    | SimpleACConfig
)

def register():
    agent_group.dispatcher(ActionScheduleAgent)
    agent_group.dispatcher(GreedyAC)
    agent_group.dispatcher(GreedyIQL)
    agent_group.dispatcher(InAC)
    agent_group.dispatcher(IQL)
    agent_group.dispatcher(RandomAgent)
    agent_group.dispatcher(SAC)
    agent_group.dispatcher(EpsilonGreedySarsa)
    agent_group.dispatcher(SimpleAC)
