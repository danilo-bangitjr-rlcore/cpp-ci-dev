from corerl.agent import agent_group, register
from corerl.agent.base import BaseAgentConfig
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState


def init_agent(cfg: BaseAgentConfig, app_state: AppState, col_desc: ColumnDescriptions):
    register()
    return agent_group.dispatch(cfg, app_state, col_desc)
