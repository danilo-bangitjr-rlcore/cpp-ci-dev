from corerl.agent.base import BaseAgent
from corerl.configs.group import Group
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.interaction.deployment_interaction import DepInteractionConfig, DeploymentInteraction
from corerl.interaction.interaction import Interaction
from corerl.interaction.sim_interaction import SimInteraction, SimInteractionConfig

interaction_group = Group[[BaseAgent, AsyncEnv, Pipeline], Interaction]()

InteractionConfig = SimInteractionConfig | DepInteractionConfig

def register():
    interaction_group.dispatcher(SimInteraction)
    interaction_group.dispatcher(DeploymentInteraction)

def init_interaction(
    cfg: InteractionConfig,
    agent: BaseAgent,
    env: AsyncEnv,
    pipeline: Pipeline,
):
    register()
    return interaction_group.dispatch(cfg, agent, env, pipeline)
