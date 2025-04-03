from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import haiku as hk
import jax
import jax.numpy as jnp

from agent.components.networks.activations import get_activation


@dataclass
class LinearConfig:
    size: int
    name: str | None = None
    activation: str = 'relu'


@dataclass
class LateFusionConfig:
    sizes: list[int]
    name: str | None = None
    activation: str = 'relu'


type LayerConfig = LinearConfig | LateFusionConfig

@dataclass
class TorsoConfig:
    layers: Sequence[LayerConfig]


class Linear(hk.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        self._linear = hk.Linear(cfg.size)

    def __call__(self, x: jax.Array):
        act = get_activation(self.cfg.activation)
        z = self._linear(x)
        return act(z)

class FusionNet(hk.Module):
    def __init__(self, cfg: LateFusionConfig):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        self.torso_branches = [
            hk.Linear(size) for size in cfg.sizes
        ]

    def __call__(self, *x: jax.Array):
        parts = [self.torso_branches[i](x[i]) for i in range(len(x))]
        z = jnp.concat(parts, axis=0)
        act = get_activation(self.cfg.activation)
        return act(z)


def layer_factory(cfg: LayerConfig):
    if isinstance(cfg, LinearConfig):
        return Linear(cfg)
    else:
        return FusionNet(cfg)

    assert_never(cfg)

def torso_builder(cfg: TorsoConfig):
    layers: list[hk.Module] = [
        layer_factory(layer_cfg)
        for layer_cfg in cfg.layers
    ]
    torso = hk.Sequential(layers)
    return torso


def ensemble_net_init(net: hk.Transformed, seed: int, ensemble: int, inputs: tuple[jax.Array, ...]):
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, ensemble)
    params = jax.vmap(net.init, in_axes=(0, None))(rngs, *inputs)
    return params
