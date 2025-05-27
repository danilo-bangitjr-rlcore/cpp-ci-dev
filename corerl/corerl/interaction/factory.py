from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.interaction.configs import InteractionConfig
from corerl.interaction.deployment_interaction import DeploymentInteraction
from corerl.interaction.sim_interaction import SimInteraction
from corerl.state import AppState


def init_interaction(
    cfg: InteractionConfig,
    app_state: AppState,
    agent: GreedyAC,
    env: DeploymentAsyncEnv,
    pipeline: Pipeline,
):
    if cfg.name == "sim_interaction":
        return SimInteraction(cfg, app_state, agent, env, pipeline)

    return DeploymentInteraction(cfg, app_state, agent, env, pipeline)
