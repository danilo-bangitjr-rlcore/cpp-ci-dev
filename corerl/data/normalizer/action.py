import gymnasium
import numpy as np
from typing import Any
from dataclasses import dataclass
from corerl.utils.hydra import Group, interpolate

from corerl.data.normalizer.base import Identity, InvertibleNormalizer, Scale


# -------------
# -- Configs --
# -------------
group = Group(
    'normalizer/action_normalizer',
    return_type=InvertibleNormalizer[np.ndarray],
)


@dataclass
class BaseNormalizerConfig:
    name: str = 'none'
    use_cfg_values: bool = False
    discrete_control: bool = interpolate('${env.discrete_control}')


@dataclass
class IdentityNormalizerConfig(BaseNormalizerConfig):
    name: str = 'identity'


@group.dispatcher
def _identity(cfg: IdentityNormalizerConfig, env: gymnasium.Env):
    return Identity()


@dataclass
class ScaleNormalizerConfig(BaseNormalizerConfig):
    name: str = 'scale'
    action_low: float = interpolate('${env.action_low}')
    action_high: float = interpolate('${env.action_high}')


@group.dispatcher
def _scale(cfg: ScaleNormalizerConfig, env: gymnasium.Env):
    action_min, action_max = get_action_bounds(cfg, env)
    return Scale(
        scale=action_max - action_min,
        bias=action_min,
    )


def init_action_normalizer(cfg: BaseNormalizerConfig, env: gymnasium.Env):
    if cfg.discrete_control and cfg.name != 'identity':
        raise Exception('Cannot normalize discrete actions')

    return group.dispatch(cfg, env)


# -----------
# -- Utils --
# -----------
def get_action_bounds(cfg: ScaleNormalizerConfig, env: gymnasium.Env) -> tuple[np.ndarray, np.ndarray]:
    if cfg.use_cfg_values:
        return (
            np.array(cfg.action_low),
            np.array(cfg.action_high),
        )

    # We don't currently have a reliable way to type-guard
    # whether the action_space has a `low` and `high`.
    space: Any = env.action_space
    return (space.low, space.high)
