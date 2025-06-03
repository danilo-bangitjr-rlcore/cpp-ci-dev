import functools
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp


@dataclass
class ActivationConfig:
    name: Any

@dataclass
class IdentityConfig(ActivationConfig):
    name: Literal['identity'] = 'identity'

def identity_act(x: jax.Array) -> jax.Array:
    return x

@dataclass
class SigmoidConfig(ActivationConfig):
    name: Literal['sigmoid'] = 'sigmoid'
    shift: float = 0.0

def sigmoid_act(cfg: SigmoidConfig, x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(x + cfg.shift)

@dataclass
class SoftsignConfig(ActivationConfig):
    name: Literal['softsign'] = 'softsign'
    shift: float = 0.0

def softsign_act(cfg: SoftsignConfig, x: jax.Array) -> jax.Array:
    return (jax.nn.soft_sign(x + cfg.shift) + 1) / 2

@dataclass
class CosineConfig(ActivationConfig):
    name: Literal['cosine'] = 'cosine'

def cos_act(cfg: CosineConfig, x: jax.Array) -> jax.Array:
    return -((jnp.cos(x) + 1) / 2)

@dataclass
class TanhConfig(ActivationConfig):
    name: Literal['tanh'] = 'tanh'
    shift: float = 0.0

def tanh_act(cfg: TanhConfig, x: jax.Array) -> jax.Array:
    return (jnp.tanh(x + cfg.shift) + 1) / 2

@dataclass
class SoftplusConfig(ActivationConfig):
    name: Literal['softplus'] = 'softplus'
    shift: float = 0.0

def softplus_act(cfg: SoftplusConfig, x: jax.Array) -> jax.Array:
    return jax.nn.softplus(x + cfg.shift) + 1

def get_output_activation(cfg: ActivationConfig):
    """
    These output activation functions return values in [0, 1]
    """
    if cfg.name == "identity":
        return identity_act
    if cfg.name == "tanh":
        assert isinstance(cfg, TanhConfig)
        return functools.partial(tanh_act, cfg)
    if cfg.name == "sigmoid":
        assert isinstance(cfg, SigmoidConfig)
        return functools.partial(sigmoid_act, cfg)
    if cfg.name in {"softsign", "soft_sign"}:
        assert isinstance(cfg, SoftsignConfig)
        return functools.partial(softsign_act, cfg)
    if cfg.name == "cosine":
        assert isinstance(cfg, CosineConfig)
        return functools.partial(cos_act, cfg)
    if cfg.name == "softplus":
        assert isinstance(cfg, SoftplusConfig)
        return functools.partial(softplus_act, cfg)
    raise NotImplementedError

def get_activation(name: str):
    if name == 'identity':
        return identity_act
    if name == 'relu':
        return jax.nn.relu
    raise NotImplementedError

def scale_shift(x: jax.Array, low: int, high: int) -> jax.Array:
    return (high - low) * x + low
