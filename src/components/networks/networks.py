import copy
import functools
from dataclasses import dataclass, field
from typing import Any, Literal

import haiku as hk
import jax
import jax.numpy as jnp
from pydantic import Field

from components.networks.activations import ActivationConfig, IdentityConfig, get_output_activation

MISSING: Any = "|???|"

def list_(vals: list[Any] | None = None) -> Any:
    if vals is None:
        return Field(default_factory=list)

    return Field(default_factory=lambda: copy.deepcopy(vals))

@dataclass
class TorsoConfig:
    name: Any = MISSING

@dataclass
class LinearTorsoConfig(TorsoConfig):
    name: Literal['linear_torso'] = 'linear_torso'
    hidden_sizes: list[int] = list_([256, 256])

class LinearTorso(hk.Module):
    def __init__(self, cfg: LinearTorsoConfig):
        super().__init__(name=cfg.name)
        self.net = hk.nets.MLP(output_sizes=list(cfg.hidden_sizes))

    def __call__(self, x: jax.Array):
        return self.net(x)

def linear_torso_fwd(cfg: LinearTorsoConfig, x: jax.Array):
    module = LinearTorso(cfg)
    return module(x)

@dataclass
class NetConfig:
    name: Any = MISSING
    output_size: int = MISSING
    output_activation: ActivationConfig = field(default_factory=IdentityConfig)

@dataclass
class LinearNetConfig(NetConfig):
    name: Literal['linear_net'] = 'linear_net'
    torso: LinearTorsoConfig = field(default_factory=LinearTorsoConfig)

class LinearNet(hk.Module):
    def __init__(self, cfg: LinearNetConfig):
        super().__init__(name=cfg.name)
        torso = LinearTorso(cfg.torso)
        output_act = get_output_activation(cfg.output_activation)
        self.net = hk.Sequential([
            torso, jax.nn.relu,
            hk.Linear(cfg.output_size), output_act
        ])

    def __call__(self, x: jax.Array):
        return self.net(x)

def linear_net_fwd(cfg: LinearNetConfig, x: jax.Array):
    module = LinearNet(cfg)
    return module(x)

@dataclass
class FusionNetConfig(NetConfig):
    name: Literal['fusion_net'] = 'fusion_net'
    branch_torso: LinearTorsoConfig = field(default_factory=LinearTorsoConfig)
    combined_torso: LinearTorsoConfig = field(default_factory=LinearTorsoConfig)

class FusionNet(hk.Module):
    def __init__(self, cfg: FusionNetConfig, input_dims: int):
        super().__init__(name=cfg.name)
        self.input_dims = input_dims
        self.torso_branches = []
        for _ in range(self.input_dims):
            self.torso_branches.append(LinearTorso(cfg.branch_torso))
        combined_torso = LinearTorso(cfg.combined_torso)
        output_act = get_output_activation(cfg.output_activation)
        self.output_net = hk.Sequential([
            combined_torso, jax.nn.relu,
            hk.Linear(cfg.output_size), output_act
        ])

    def __call__(self, x: jax.Array):
        # TODO: vmap?
        branch_out = [self.torso_branches[i](x[i]) for i in range(self.input_dims)]
        branch_out_cat = jnp.concat(branch_out)
        return self.output_net(branch_out_cat)

def fusion_net_fwd(cfg: FusionNetConfig, input_dims: int, x: jax.Array):
    module = FusionNet(cfg, input_dims)
    return module(x)

def network_init(cfg: NetConfig, input_dims: int):
    if cfg.name == "linear_net":
        assert isinstance(cfg, LinearNetConfig)
        linear_net_fwd_fn = functools.partial(linear_net_fwd, cfg)
        linear_net = hk.without_apply_rng(hk.transform(linear_net_fwd_fn))
        return linear_net
    elif cfg.name == "fusion_net":
        assert isinstance(cfg, FusionNetConfig)
        fusion_net_fwd_fn = functools.partial(fusion_net_fwd, cfg, input_dims)
        fusion_net = hk.without_apply_rng(hk.transform(fusion_net_fwd_fn))
        return fusion_net
    else:
        raise NotImplementedError

@dataclass
class EnsembleNetConfig:
    subnet: NetConfig = field(default_factory=FusionNetConfig)
    ensemble: int = 1

def ensemble_net_init(cfg: EnsembleNetConfig, seed: int, input_dims: int, x: jax.Array):
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, cfg.ensemble)
    sub_net = network_init(cfg.subnet, input_dims)
    params = jax.vmap(sub_net.init, in_axes=(0, None))(rngs, x)

    return params

def ensemble_net_fwd(cfg: EnsembleNetConfig, input_dims: int, params: dict, x: jax.Array):
    sub_net = network_init(cfg.subnet, input_dims)
    outputs = jax.vmap(sub_net.apply, in_axes=(0, None))(params, x)

    return outputs
