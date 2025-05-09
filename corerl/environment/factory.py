import logging
from typing import Any

import gymnasium as gym
from coreenv.pertube_env import PerturbationConfig

from corerl.environment.async_env.async_env import GymEnvConfig

try:
    from coreenv.factory import init_env
except ImportError:

    def init_env(name: str, overrides: Any = None, perturbation_config: PerturbationConfig | None = None) -> Any:
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
            return gym.make(cfg.gym_name, *args, **kwargs)
        case "custom":
            return init_env(cfg.gym_name, overrides=cfg.env_config, perturbation_config=cfg.perturb_config)
        case _:
            raise NotImplementedError
