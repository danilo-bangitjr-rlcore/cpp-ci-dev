import logging

import gymnasium as gym
from corerl.environment.async_env.async_env import GymEnvConfig
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig
from corerl.environment.model_env import ModelEnv, ModelEnvConfig

log = logging.getLogger(__name__)

def init_environment(cfg: GymEnvConfig) -> gym.Env:
    args = cfg.args
    kwargs = cfg.kwargs
    if isinstance(cfg, DepAsyncEnvConfig) and (bool(args) or bool(kwargs)):
        log.warning("Environment args or kwargs set in confinguration for DeploymentAsyncEnv. This will be ignored!")
        args = []
        kwargs = {}

    match cfg.init_type:
        case 'gym.make':
            if cfg.env_config is not None:
                kwargs = dict(kwargs)
                kwargs['cfg'] = cfg.env_config
            return gym.make(cfg.gym_name, *args, **kwargs)
        case 'custom':
            return init_custom_env(cfg)
        case _:
            raise NotImplementedError


def init_custom_env(cfg: GymEnvConfig) -> gym.Env:
    name = cfg.gym_name

    match name:
        case 'model_env':
            assert isinstance(cfg, ModelEnvConfig)
            return ModelEnv(cfg)
        case _:
            raise NotImplementedError
