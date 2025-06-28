from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.async_env.sim_async_env import SimAsyncEnv
from corerl.tags.tag_config import TagConfig


def init_async_env(cfg: AsyncEnvConfig, tag_config: list[TagConfig]):
    if cfg.name == "sim_async_env":
        return SimAsyncEnv(cfg, tag_config)
    return DeploymentAsyncEnv(cfg, tag_config)
