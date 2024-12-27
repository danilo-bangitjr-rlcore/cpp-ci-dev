from corerl.configs.group import Group
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.async_env.sim_async_env import SimAsyncEnv, SimAsyncEnvConfig
from corerl.environment.async_env.tsdb_async_stub_env import TSDBAsyncStubEnv, TSDBAsyncStubEnvConfig
from corerl.environment.async_env.opc_tsdb_sim_async_env import OPCTSDBSimAsyncEnv, OPCTSDBSimAsyncEnvConfig

async_env_group = Group[
    [list[TagConfig]],
    AsyncEnv
]()

AsyncEnvConfig = (
    SimAsyncEnvConfig
    | TSDBAsyncStubEnvConfig
    | OPCTSDBSimAsyncEnvConfig
)


def register():
    async_env_group.dispatcher(SimAsyncEnv)
    async_env_group.dispatcher(TSDBAsyncStubEnv)
    async_env_group.dispatcher(OPCTSDBSimAsyncEnv)


def init_async_env(cfg: AsyncEnvConfig, tag_config: list[TagConfig]):
    register()
    return async_env_group.dispatch(cfg, tag_config)

