from corerl.configs.environment.async_env import AsyncEnvConfig
from corerl.configs.tags.tag_config import TagConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.async_env.sim_async_env import SimAsyncEnv


def init_async_env(cfg: AsyncEnvConfig, tag_config: list[TagConfig]):
    if cfg.name == "sim_async_env":
        return SimAsyncEnv(cfg, tag_config)
    return DeploymentAsyncEnv(cfg, tag_config)
