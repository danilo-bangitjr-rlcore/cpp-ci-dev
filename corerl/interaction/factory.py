from typing import Annotated

from pydantic import Field

from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.group import Group
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.interaction.deployment_interaction import DepInteractionConfig, DeploymentInteraction
from corerl.interaction.interaction import Interaction
from corerl.interaction.sim_interaction import SimInteraction, SimInteractionConfig
from corerl.state import AppState

interaction_group = Group[[AppState, GreedyAC, AsyncEnv, Pipeline], Interaction]()


InteractionConfig = Annotated[
    SimInteractionConfig
    | DepInteractionConfig,
    Field(discriminator='name')
]

def register():
    interaction_group.dispatcher(SimInteraction)
    interaction_group.dispatcher(DeploymentInteraction)

def init_interaction(
    cfg: InteractionConfig,
    app_state: AppState,
    agent: GreedyAC,
    env: AsyncEnv,
    pipeline: Pipeline,
):
    register()
    return interaction_group.dispatch(cfg, app_state, agent, env, pipeline)
