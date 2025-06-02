import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Concatenate, Protocol, TypeVar

from rl_env.pertube_env import ObservationPerturbationWrapper, PerturbationConfig

logger = logging.getLogger(__name__)

MISSING: Any = "|???|"

@dataclass
class EnvConfig:
    name : str = MISSING
    seed : int = 0

class DiscriminatedUnion(Protocol):
    @property
    def name(self) -> Any: ...


Config = TypeVar('Config', bound=DiscriminatedUnion)


class Group[**P, R]:
    def __init__(self):
        self._dispatchers: dict[str, Callable[..., R]] = {}
        self._cfgs: dict[str, EnvConfig] = {}

    def dispatcher(
        self,
        cfg: EnvConfig,
        f: Callable[Concatenate[Config, P], R],
    ):
        self._dispatchers[cfg.name] = f
        self._cfgs[cfg.name] = cfg

        return f

    def dispatch(self, name: str, overrides: dict | None = None):
        if overrides is None:
            overrides = {}
        cfg = self._cfgs[name]
        cfg = replace(cfg, **overrides)
        return self._dispatchers[name](cfg)

env_group = Group[[], Any]()

def init_env(name: str, overrides: dict | None = None, perturbation_config: PerturbationConfig | None = None):
    # register environments
    import rl_env.calibration
    import rl_env.distraction_world
    import rl_env.four_rooms
    import rl_env.multi_action_saturation
    import rl_env.pvs
    import rl_env.saturation
    import rl_env.saturation_goals
    import rl_env.stand_still_mc
    import rl_env.t_maze
    import rl_env.three_tanks
    import rl_env.windy_room  # noqa: F401

    logger.info(f"instantiating {name} with overrides {overrides}")

    if name == 'DelayedSaturation-v0':
        warnings.warn(
            "Delayed-Saturation-v0 is deprecated. Use Saturation-v0 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = 'Saturation-v0'
        overrides = {
            'decay': 0.75,
            'effect': None,
            'effect_period': 500,
            'trace_val': 0.9,
        }

    env = env_group.dispatch(name, overrides)
    if perturbation_config:
        print("perturbation_config", perturbation_config)
        env = ObservationPerturbationWrapper(env, perturbation_config)
    return env
