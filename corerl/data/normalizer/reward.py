from dataclasses import dataclass
from corerl.utils.hydra import Group, interpolate, DiscriminatedUnion

from corerl.data.normalizer.base import BaseNormalizer, Clip, Identity, Scale


group = Group[
    [],
    BaseNormalizer[float],
]('normalizer/reward_normalizer')


@dataclass
class IdentityNormalizerConfig:
    name: str = 'identity'


@group.dispatcher
def _identity(cfg: IdentityNormalizerConfig):
    return Identity()


@dataclass
class ScaleNormalizerConfig:
    name: str = 'scale'
    reward_low: float = interpolate('${env.reward_low}')
    reward_high: float = interpolate('${env.reward_high}')
    reward_bias: float = 0


@group.dispatcher
def _scale(cfg: ScaleNormalizerConfig):
    return Scale(
        scale=cfg.reward_high - cfg.reward_low,
        bias=cfg.reward_bias,
    )


@dataclass
class ClipNormalizerConfig:
    name: str = 'clip'
    clip_min: float = -2
    clip_max: float = 1


@group.dispatcher
def _clip(cfg: ClipNormalizerConfig):
    return Clip(cfg.clip_min, cfg.clip_max)


def init_reward_normalizer(cfg: DiscriminatedUnion):
    return group.dispatch(cfg)
