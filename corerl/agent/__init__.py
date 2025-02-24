from typing import Annotated

from pydantic import Field

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC, GreedyACConfig
from corerl.configs.group import Group
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState

agent_group = Group[
    [AppState, ColumnDescriptions],
    BaseAgent,
]()

AgentConfig = Annotated[
    GreedyACConfig
, Field(discriminator='name')]


def register():
    agent_group.dispatcher(GreedyAC)
