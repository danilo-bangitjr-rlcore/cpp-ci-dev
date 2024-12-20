from corerl.configs.group import Group

from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.async_env.sim_async_env import SimAsyncEnv, SimAsyncEnvConfig
from corerl.environment.async_env.tsdb_async_stub_env import TSDBAsyncStubEnvConfig, TSDBAsyncStubEnv
from corerl.data_pipeline.tag_config import TagConfig


async_env_group = Group[
    [list[TagConfig]],
    AsyncEnv
]()

AsyncEnvConfig = (
    SimAsyncEnvConfig
    | TSDBAsyncStubEnvConfig
)


def register():
    async_env_group.dispatcher(SimAsyncEnv)
    async_env_group.dispatcher(TSDBAsyncStubEnv)


def init_async_env(cfg: AsyncEnvConfig, tag_config: list[TagConfig]):
    register()
    return async_env_group.dispatch(cfg, tag_config)

