import gymnasium
import numpy as np
from typing import Any
from dataclasses import dataclass
from corerl.utils.hydra import Group, interpolate

from corerl.data.normalizer.base import AvgNanNorm, Identity, InvertibleNormalizer, MaxMin


# -------------
# -- Configs --
# -------------
group = Group[
    [gymnasium.Env],
    InvertibleNormalizer[np.ndarray],
]('normalizer/obs_normalizer')


@dataclass
class BaseNormalizerConfig:
    name: str = 'none'
    discrete_control: bool = interpolate('${env.discrete_control}')


@dataclass
class IdentityNormalizerConfig(BaseNormalizerConfig):
    name: str = 'identity'


@group.dispatcher
def _identity(cfg: IdentityNormalizerConfig, env: gymnasium.Env):
    return Identity()


@dataclass
class MaxminNormalizerConfig(BaseNormalizerConfig):
    name: str = 'maxmin'


@group.dispatcher
def _maxmin(cfg: MaxminNormalizerConfig, env: gymnasium.Env):
    lo, hi = get_observation_bounds(env)
    return MaxMin(lo, hi)


@dataclass
class AvgNanNormalizerConfig(BaseNormalizerConfig):
    name: str = 'avg_nan_norm'


@group.dispatcher
def _avgnan(cfg: AvgNanNormalizerConfig, env: gymnasium.Env):
    lo, hi = get_observation_bounds(env)
    return AvgNanNorm(lo, hi)


def init_obs_normalizer(cfg: BaseNormalizerConfig, env: gymnasium.Env):
    return group.dispatch(cfg, env)


# -----------
# -- Utils --
# -----------
def get_observation_bounds(env: gymnasium.Env):
    # We don't currently have a reliable way to type-guard
    # whether the observation_space has a `low` and `high`.
    space: Any = env.observation_space

    return (
        space.low,
        space.high,
    )
