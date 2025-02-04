from corerl.configs.group import Group
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig, DeploymentAsyncEnv
from corerl.environment.async_env.sim_async_env import SimAsyncEnv, SimAsyncEnvConfig

async_env_group = Group[
    [list[TagConfig]],
    AsyncEnv
]()

AsyncEnvConfig = (
    SimAsyncEnvConfig
    | DepAsyncEnvConfig
)


def register():
    async_env_group.dispatcher(SimAsyncEnv)
    async_env_group.dispatcher(DeploymentAsyncEnv)


def init_async_env(cfg: AsyncEnvConfig, tag_config: list[TagConfig]):
    register()
    return async_env_group.dispatch(cfg, tag_config)
