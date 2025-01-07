import gymnasium as gym
import logging

from corerl.environment.async_env.async_env import BaseAsyncEnvConfig
from corerl.environment.config import EnvironmentConfig
from corerl.environment.model_env import ModelEnv, ModelEnvConfig
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig

log = logging.getLogger(__name__)


def init_environment(cfg: EnvironmentConfig) -> gym.Env:
    args = cfg.args
    kwargs = cfg.kwargs
    if isinstance(cfg, DepAsyncEnvConfig) and (bool(args) or bool(kwargs)):
        log.warning("Environment args or kwargs set in confinguration for DeploymentAsyncEnv. This will be ignored!")
        args = []
        kwargs = {}

    match cfg.type:
        case 'gym.make':
            assert isinstance(cfg, BaseAsyncEnvConfig)
            return gym.make(cfg.gym_name, *args, **kwargs)
        case 'custom':
            return init_custom_env(cfg)
        case _:
            raise NotImplementedError


def init_custom_env(cfg: EnvironmentConfig) -> gym.Env:
    name = cfg.name

    match name:
        case 'model_env':
            assert isinstance(cfg, ModelEnvConfig)
            return ModelEnv(cfg)
        case _:
            raise NotImplementedError
