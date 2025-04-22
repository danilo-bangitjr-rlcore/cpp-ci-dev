import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Concatenate, Protocol, TypeVar

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

def init_env(name: str, overrides: dict | None = None):
    # register environments
    import coreenv.calibration  # noqa: F401
    import coreenv.distraction_world  # noqa: F401
    import coreenv.four_rooms  # noqa: F401
    import coreenv.multi_action_saturation  # noqa: F401
    import coreenv.pvs  # noqa: F401
    import coreenv.saturation  # noqa: F401
    import coreenv.saturation_goals  # noqa: F401
    import coreenv.stand_still_mc  # noqa: F401
    import coreenv.three_tanks  # noqa: F401
    import coreenv.windy_room  # noqa: F401

    logger.info(f"instantiaing {name} with overrides {overrides}")

    if name == 'DelayedSaturation-v0':
        warnings.warn(
            "Delayed-Saturation-v0 is deprecated. Use Saturation-v0 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        name = 'Saturation-v0'
        overrides = {
            'decay' : 0.75,
            'effect' : None,
            'effect_period' : 500,
            'trace_val' : 0.9
        }


    return env_group.dispatch(name, overrides)
