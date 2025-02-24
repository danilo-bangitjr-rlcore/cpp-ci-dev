from typing import Annotated

from pydantic import Field

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC, GreedyACConfig
from corerl.agent.sarsa import EpsilonGreedySarsa, EpsilonGreedySarsaConfig
from corerl.agent.simple_ac import SimpleAC, SimpleACConfig
from corerl.configs.group import Group
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState

agent_group = Group[
    [AppState, ColumnDescriptions],
    BaseAgent,
]()

AgentConfig = Annotated[
    GreedyACConfig
    | EpsilonGreedySarsaConfig
    | SimpleACConfig
, Field(discriminator='name')]


def register():
    agent_group.dispatcher(GreedyAC)
    agent_group.dispatcher(EpsilonGreedySarsa)
    agent_group.dispatcher(SimpleAC)
