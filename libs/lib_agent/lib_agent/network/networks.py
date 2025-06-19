from collections.abc import Sequence
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from lib_agent.network.activations import get_activation


class CallableModule(hk.Module):
    def __call__(self, *args: jax.Array):
        raise NotImplementedError


@dataclass
class LinearConfig:
    size: int
    name: str | None = None
    activation: str = 'crelu'

@dataclass
class ResidualConfig:
    size: int
    name: str | None = None
    activation: str = 'crelu'

@dataclass
class LateFusionConfig:
    streams: list[list[LinearConfig | ResidualConfig]]
    name: str | None = None

type LayerConfig = LinearConfig | LateFusionConfig | ResidualConfig

@dataclass
class TorsoConfig:
    layers: Sequence[LayerConfig]
    skip: bool = False


class Linear(hk.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        ortho = hk.initializers.Orthogonal(np.sqrt(2))
        self._linear = hk.Linear(
            self.cfg.size,
            w_init=ortho,
        )

    def __call__(self, x: jax.Array):
        act = get_activation(self.cfg.activation)
        z = self._linear(x)
        return act(z)

class FusionNet(hk.Module):
    def __init__(self, cfg: LateFusionConfig):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        self.torso_branches = [
            [
                layer_factory(layer)
                for layer in stream
            ]
            for stream in cfg.streams
        ]

    def __call__(self, *x: jax.Array):
        streams: list[jax.Array] = []
        for input, stream in zip(x, self.torso_branches, strict=True):
            stream_out = input
            for layer in stream:
                stream_out = layer(stream_out)
            streams.append(stream_out)

        return jnp.concat(streams, axis=-1)


class ResidualBlock(hk.Module):
    def __init__(self, cfg: ResidualConfig, output_size: int | None=None):
        super().__init__(name=cfg.name)
        if output_size is None:
            output_size = cfg.size

        self.activation = cfg.activation
        projection_size = output_size if cfg.activation not in {'crelu'} else output_size * 2

        ortho = hk.initializers.Orthogonal(np.sqrt(2))
        self.linear = hk.Linear(output_size, w_init=ortho)
        self.projection = hk.Linear(projection_size, w_init=ortho)

    def __call__(self, x: jax.Array):
        out = self.linear(x)
        out = get_activation(self.activation)(out)
        return out + self.projection(x)


class SkipProjNet(hk.Module):
    def __init__(self, torso: CallableModule, num_inputs: int, out_size: int):
        super().__init__()
        self.torso = torso
        ortho = hk.initializers.Orthogonal(np.sqrt(2))
        self.skips = [hk.Linear(out_size, w_init=ortho) for _ in range(num_inputs)]

    def __call__(self, *x: jax.Array):
        out = self.torso(*x)
        skip_outs = [self.skips[i](y) for i, y in enumerate(x)]

        for skip_out in skip_outs:
            out += skip_out

        return out


def layer_factory(cfg: LayerConfig):
    if isinstance(cfg, LinearConfig):
        return Linear(cfg)
    if isinstance(cfg, ResidualConfig):
        return ResidualBlock(cfg)

    return FusionNet(cfg)

def torso_builder(cfg: TorsoConfig):
    layers: list[hk.Module] = [
        layer_factory(layer_cfg)
        for layer_cfg in cfg.layers
    ]
    torso = hk.Sequential(layers)

    if cfg.skip:
        last_layer = cfg.layers[-1]
        assert isinstance(last_layer, LinearConfig | ResidualConfig)
        out_size = last_layer.size
        out_size = out_size * 2 if last_layer.activation in {'crelu'} else out_size

        input_layer = cfg.layers[0]
        assert isinstance(input_layer, LateFusionConfig)

        num_inputs = len(input_layer.streams)
        torso = SkipProjNet(torso, num_inputs, out_size)

    return torso


def ensemble_net_init(net: hk.Transformed, seed: int, ensemble: int, inputs: tuple[jax.Array, ...]):
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, ensemble)
    return jax.vmap(net.init, in_axes=(0, None))(rngs, *inputs)
