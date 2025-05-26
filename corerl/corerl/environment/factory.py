import logging
from typing import Any

import gymnasium as gym

from corerl.environment.async_env.async_env import GymEnvConfig
from corerl.environment.wrapper.wrappers import wrappers

try:
    from coreenv.factory import init_env
except ImportError:

    def init_env(name: str, overrides: Any = None) -> Any:
        """
        Placeholder function for initializing a custom environment.
        This should be replaced with the actual implementation.
        """
        raise NotImplementedError("Custom environment initialization not implemented.")


log = logging.getLogger(__name__)


def init_environment(cfg: GymEnvConfig) -> gym.Env:
    args = cfg.args
    kwargs = cfg.kwargs

    match cfg.init_type:
        case "gym.make":
            if cfg.env_config is not None:
                kwargs = dict(kwargs)
                kwargs["cfg"] = cfg.env_config
            env = gym.make(cfg.gym_name, *args, **kwargs)
        case "custom":
            env = init_env(cfg.gym_name, overrides=cfg.env_config)
        case _:
            raise NotImplementedError

    if cfg.wrapper.name is not None:
        env = wrappers[cfg.wrapper.name](env, **cfg.wrapper.kwargs)
    return env
