import logging
from typing import Any

from corerl.configs.environment.async_env import GymEnvConfig

try:
    import gymnasium as gym
    from rl_env.factory import init_env
except ImportError:

    def init_env(cfg: Any, perturbation_config: dict | None = None) -> Any:
        """
        Placeholder function for initializing a custom environment.
        This should be replaced with the actual implementation.
        """
        raise NotImplementedError("Custom environment initialization not implemented.")

    gym = None


log = logging.getLogger(__name__)


def init_environment(cfg: GymEnvConfig):
    from corerl.environment.wrapper.wrappers import wrappers

    match cfg.init_type:
        case "custom":
            env = init_env(cfg.env_config, perturbation_config=cfg.perturb_config)
        case _:
            raise NotImplementedError

    if cfg.wrapper.name is not None:
        env = wrappers[cfg.wrapper.name](env, **cfg.wrapper.kwargs)
    return env
