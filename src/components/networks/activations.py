import functools
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Literal

MISSING: Any = "|???|"

@dataclass
class ActivationConfig:
    name: Any = MISSING

@dataclass
class IdentityConfig(ActivationConfig):
    name: Literal['identity'] = 'identity'

def identity_act(x):
    return x

@dataclass
class SigmoidConfig(ActivationConfig):
    name: Literal['sigmoid'] = 'sigmoid'
    shift: float = 0.0

def sigmoid_act(cfg: SigmoidConfig, x):
    return jax.nn.sigmoid(x + cfg.shift)

@dataclass
class SoftsignConfig(ActivationConfig):
    name: Literal['softsign'] = 'softsign'
    shift: float = 0.0

def softsign_act(cfg: SoftsignConfig, x):
    return (jax.nn.soft_sign(x + cfg.shift) + 1) / 2

@dataclass
class CosineConfig(ActivationConfig):
    name: Literal['cosine'] = 'cosine'

def cos_act(cfg: CosineConfig, x):
    return -((jnp.cos(x) + 1) / 2)

@dataclass
class TanhConfig(ActivationConfig):
    name: Literal['tanh'] = 'tanh'
    shift: float = 0.0

def tanh_act(cfg: TanhConfig, x):
    return (jnp.tanh(x + cfg.shift) + 1) / 2

def get_output_activation(cfg: ActivationConfig):
    """
    These output activation functions return values in [0, 1]
    """
    if cfg.name == "identity":
        return identity_act
    elif cfg.name == "tanh":
        tanh_act_fn = functools.partial(tanh_act, cfg)
        return tanh_act_fn
    elif cfg.name == "sigmoid":
        sigmoid_act_fn = functools.partial(sigmoid_act, cfg)
        return sigmoid_act_fn
    elif cfg.name == "softsign" or cfg.name == "soft_sign":
        softsign_act_fn = functools.partial(softsign_act, cfg)
        return softsign_act_fn
    elif cfg.name == "cosine":
        cosine_act_fn = functools.partial(cos_act, cfg)
        return cosine_act_fn
    else:
        raise NotImplementedError