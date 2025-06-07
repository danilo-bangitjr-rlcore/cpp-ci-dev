import logging
from typing import Any

from corerl.environment.async_env.async_env import GymEnvConfig

try:
    import gymnasium as gym
    from rl_env.factory import init_env
except ImportError:

    def init_env(name: str, overrides: Any = None, perturbation_config: Any | None = None) -> Any:
        """
        Placeholder function for initializing a custom environment.
        This should be replaced with the actual implementation.
        """
        raise NotImplementedError("Custom environment initialization not implemented.")

    gym = None


log = logging.getLogger(__name__)


def init_environment(cfg: GymEnvConfig):
    from corerl.environment.wrapper.wrappers import wrappers

    args = cfg.args
    kwargs = cfg.kwargs

    match cfg.init_type:
        case "gym.make":
            if gym is None:
                raise ImportError("Gymnasium is not installed. Please install it to use this functionality.")

            if cfg.env_config is not None:
                kwargs = dict(kwargs)
                kwargs["cfg"] = cfg.env_config
            env = gym.make(cfg.gym_name, *args, **kwargs)
        case "custom":
            env = init_env(cfg.gym_name, overrides=cfg.env_config, perturbation_config=cfg.perturb_config)
        case _:
            raise NotImplementedError

    if cfg.wrapper.name is not None:
        env = wrappers[cfg.wrapper.name](env, **cfg.wrapper.kwargs)
    return env
