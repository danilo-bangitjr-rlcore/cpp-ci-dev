import corerl.component.network.utils as utils

from abc import ABC, abstractmethod
from warnings import warn
from typing import Union
import torch
import torch.distributions as d
import torch.nn as nn


_HalfBoundedConstraint = Union[
    d.constraints.greater_than_eq,
    d.constraints.greater_than,
    d.constraints.less_than,
]

_BoundedAboveConstraint = Union[
    d.constraints.less_than,
]

_BoundedBelowConstraint = Union[
    d.constraints.greater_than_eq,
    d.constraints.greater_than,
]


class ArctanhNormal(d.Distribution):
    _EPSILON = 1e-6

    arg_constraints = {
        "loc": d.constraints.real, "scale": d.constraints.positive,
    }
    support = d.constraints.interval(-1.0, 1.0)
    has_rsample = True

    @property
    def mean(self):
        return torch.tanh(self.loc)

    @property
    def loc(self):
        return self._underlying.loc

    @loc.setter
    def loc(self, value):
        self._underlying.loc = value

    @property
    def scale(self):
        return self._underlying.scale

    @scale.setter
    def scale(self, value):
        self._underlying.scale = value

    @property
    def batch_shape(self):
        return self._underlying.batch_shape

    @property
    def event_shape(self):
        return self._underlying.event_shape

    def __init__(self, loc, scale, validate_args=None):
        self._underlying = d.Normal(loc, scale, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ArctanhNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(ArctanhNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        samples = self._underlying.sample(sample_shape=sample_shape)
        return torch.tanh(samples)

    def rsample(self, sample_shape=torch.Size()):
        samples = self._underlying.rsample(sample_shape=sample_shape)
        return torch.tanh(samples)

    def log_prob(self, value):
        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_logp = self._underlying.log_prob(normal_samples)

        offset = torch.log1p(
            -value.pow(2) + ArctanhNormal._EPSILON
        )

        return normal_logp - offset

    def cdf(self, value):
        less_than_mask = torch.where(value <= -1)
        greater_than_mask = torch.where(value >= 1)

        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_cdf = self._underlying.cdf(normal_samples)
        normal_cdf[less_than_mask] = 0
        normal_cdf[greater_than_mask] = 1

        return normal_cdf

    def icdf(self, value):
        less_than_mask = torch.where(value <= -1)
        greater_than_mask = torch.where(value >= 1)

        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_cdf = self._underlying.icdf(normal_samples)
        normal_cdf[less_than_mask] = 1
        normal_cdf[greater_than_mask] = 0

        return normal_cdf

    def entropy(self):
        """
        Note that this function does not return the entropy of the
        ArctanhNormal distribution, which does not have a closed-form solution
        for the entropy function.

        Instead, this function enables the gradient of the distribution entropy
        to be correctly computed.
        """
        return self._underlying.entropy()


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self._min = min
        self._max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self._min, self._max)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TanhShift(nn.Module):
    def __init__(self, shift, denom, high, low):
        super().__init__()
        self._shift = shift
        self._denom = denom

        self._high = high
        self._low = low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_out = torch.tanh((x + self._shift * self._denom) / self._denom)
        normalized = (tanh_out + 1) / 2
        return normalized * (self._high - self._low) + self._low


def init_activation(cfg) -> nn.Module:
    name = cfg["name"]
    args = cfg.get("args", tuple())
    kwargs = cfg.get("kwargs", {})

    activations = {
        "exp": Exp,
        "clamp": Clamp,
        "tanh_shift": TanhShift,
        "none": Identity,
        "identity": Identity,
        None: Identity,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "prelu": torch.nn.PReLU,
        "rrelu": torch.nn.RReLU,
        "celu": torch.nn.CELU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "softplus": torch.nn.Softplus,
        "softmax": torch.nn.Softmax,
        "softmin": torch.nn.Softmin,
        "sigmoid": torch.nn.Sigmoid,
        "log_sigmoid": torch.nn.LogSigmoid,
        "hard_sigmoid": torch.nn.Hardsigmoid,
        "elu": torch.nn.ELU,
        "hard_tanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
    }

    if name.lower() not in activations.keys():
        raise NotImplementedError(
            f"unknown activation function {name}, known activations include " +
            f"{list(activations.keys())}",
        )

    return activations[name.lower()](*args, **kwargs)


class Parallel(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self._paths = nn.ModuleList(layers)

    def forward(self, x):
        return tuple(path(x) for path in self._paths)

    def string(self):
        return f"Parallel[{self._paths}]"


class Policy(ABC):
    def __init__(self, model):
        self._model = model

    @classmethod
    @abstractmethod
    def from_env(cls, model, dist, env):
        pass

    @abstractmethod
    def _transform_from_params(self, *params):
        pass

    @abstractmethod
    def _transform(self, dist):
        pass

    @property
    @abstractmethod
    def param_names(self):
        pass

    @property
    def n_params(self):
        return len(self.param_names)

    @property
    @abstractmethod
    def support(self):
        pass

    def forward(self, state, rsample=True):
        params = self._model(state)
        dist = self._transform_from_params(*params)

        info = dict(zip(
            [param_name for param_name in self.param_names],
            [p.squeeze().detach().numpy() for p in params]
        ))

        if rsample:
            samples = dist.rsample()
        else:
            samples = dist.sample()

        return samples, info

    def log_prob(self, state: torch.Tensor, action: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)

        if not torch.all(dist.support.check(action)):
            raise ValueError(
                "expected all actions to be within the distribution support " +
                f"of {dist.support}, but got actions: \n{action}"
            )

        lp = dist.log_prob(action)
        lp = lp.view(-1, 1)

        return lp, None

    def mean(self, state: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.mean

    def mode(self, state: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.mode

    def variance(self, state: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.variance

    def stddev(self, state: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.stddev

    def entropy(self, state: torch.Tensor):
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.entropy()

    def kl(self, other, state):
        self_params = self._model(state)
        self_dist = self._transform_from_params(*self_params)

        other_params = other._model(state)
        other_dist = other._transform_from_params(*other_params)

        return d.kl.kl_divergence(self_dist, other_dist)


class HalfBounded(Policy):
    """
    HalfBounded is a policy on a half-bounded support interval, either
    `(action_min, ∞)` or `(-∞, action_max)`.

    If given a distribution `dist` which has support unbounded above, then
    `dist` will be transformed to cover the action space, such that `dist` has
    support `(action_min, ∞)`.

    If given a distribution `dist` which has support unbounded below, then
    `dist` will be transformed to cover the action space, such that `dist` has
    support `(-∞, action_max)`.
    """
    def __init__(self, model, dist, action_min=None, action_max=None):
        super().__init__(model)
        self._dist = dist

        if isinstance(dist.support, _BoundedAboveConstraint):
            self._dist_max = dist.support.upper_bound
            self._dist_min = -torch.inf
            if action_min is not None and action_min != -torch.inf:
                warn(
                    f"the support of {dist} is not bounded below, ignoring" +
                    "action_min value {action_min}"
                )
            action_min = -torch.inf

            if action_max is None:
                action_max = self._dist.support.upper_bound
            assert action_max != torch.inf

        elif isinstance(dist.support, _BoundedBelowConstraint):
            self._dist_min = dist.support.lower_bound
            self._dist_max = torch.inf
            if action_max is not None and action_max != torch.inf:
                warn(
                    f"the support of {dist} is not bounded above, ignoring " +
                    f"action_max value {action_max}"
                )
            action_max = torch.inf

            if action_min is None:
                action_min = self._dist.support.lower_bound
            assert action_min != -torch.inf

        else:
            raise ValueError(
                "HalfBounded expects dist to have support type " +
                f"{_HalfBoundedConstraint}, but " +
                f"got {type(dist.support)}"
            )

        assert action_min < action_max
        self._action_min = action_min
        self._action_max = action_max

    @property
    def support(self):
        if self._bounded_below:
            lb = self._dist.support.lower_bound
            return type(self._dist.support)(lb + self._action_min)
        else:
            ub = self._dist.support.upper_bound
            return type(self._dist.support)(ub + self._action_max)

    @property
    def _bounded_below(self):
        return self._dist_min is not -torch.inf

    @property
    def _bounded_above(self):
        return self._dist_max is not torch.inf

    @property
    def param_names(self):
        return tuple(self._dist.arg_constraints.keys())

    @classmethod
    def from_env(cls, model, dist, env):
        action_min = torch.Tensor(env.action_space.low)
        action_max = torch.Tensor(env.action_space.high)

        return cls(model, dist, action_min, action_max)

    def _transform_from_params(self, *params):
        return self._transform(self._dist(*params))

    def _transform(self, dist):
        if self._bounded_below:
            transform = d.AffineTransform(loc=self._action_min, scale=1)
            dist = d.TransformedDistribution(dist, [transform])
        else:
            transform = d.AffineTransform(loc=self._action_max, scale=1)

        dist = d.TransformedDistribution(dist, [transform])

        return d.Independent(dist, 1)

    def __repr__(self):
        return f"HalfBounded[{self._dist}, {self._model}]"


class Bounded(Policy):
    def __init__(self, model, dist, action_min=None, action_max=None):
        super().__init__(model)
        self._dist = dist

        if not isinstance(dist.support, d.constraints.interval):
            raise ValueError(
                "Bounded expects dist to have support type " +
                f"{d.constraints.interval}, but " +
                f"got {type(dist.support)}"
            )
        dist_min = dist.support.lower_bound
        dist_max = dist.support.upper_bound

        if action_min is None:
            action_min = self._dist.support.lower_bound
        if action_max is None:
            action_max= self._dist.support.upper_bound

        assert action_min < action_max

        self._action_scale = (action_max - action_min) / (dist_max - dist_min)
        self._action_bias = -self._action_scale * dist_min + action_min

    @property
    def support(self):
        lb = self._dist.support.lower_bound
        ub = self._dist.support.upper_bound

        return type(self._dist.support)(
            lower_bound=self._action_scale * lb + self._action_bias,
            upper_bound=self._action_scale * ub + self._action_bias,
        )

    @property
    def param_names(self):
        return tuple(self._dist.arg_constraints.keys())

    @classmethod
    def from_env(cls, model, dist, env):
        action_min = torch.Tensor(env.action_space.low)
        action_max = torch.Tensor(env.action_space.high)

        return cls(model, dist, action_min, action_max)

    def _transform_from_params(self, *params):
        return self._transform(self._dist(*params))

    def _transform(self, dist):
        if self._action_bias != 0 or self._action_scale != 1:
            transform = d.AffineTransform(
                loc=self._action_bias, scale=self._action_scale,
            )
            dist = d.TransformedDistribution(dist, [transform])

        return d.Independent(dist, 1)

    def __repr__(self):
        return f"Bounded[{self._dist}, {self._model}]"


class UnBounded(Policy):
    def __init__(self, model, dist):
        super().__init__(model)
        self._dist = dist

        if not isinstance(dist.support, type(d.constraints.real)):
            raise ValueError(
                "UnBounded expects dist to have support type " +
                f"{type(d.constraints.real)}, but " +
                f"got {type(dist.support)}"
            )

    @property
    def support(self):
        return self._dist.support

    @property
    def param_names(self):
        return tuple(self._dist.arg_constraints.keys())

    @classmethod
    def from_env(cls, model, dist, env):
        return cls(model, dist)

    def _transform_from_params(self, *params):
        return self._transform(self._dist(*params))

    def _transform(self, dist):
        return d.Independent(dist, 1)

    def __repr__(self):
        return f"UnBounded[{self._dist}, {self._model}]"





def get_policy_type(type_):
    if type_.lower() == "bounded":
        return Bounded
    elif type_.lower() == "unbounded":
        return UnBounded
    elif type_.lower() == "halfbounded":
        return HalfBounded
    else:
        raise NotImplementedError(f"unknown policy type {type_}")


def get_dist_type(type_):
    if type_.lower() in ("arctanhnormal", "squashed_gaussian"):
        return ArctanhNormal
    elif type_.lower() == "beta":
        return d.Beta
    elif type_.lower() == "logitnormal":
        return d.LogitNormal
    elif type_.lower() == "gamma":
        return d.Gamma
    elif type_.lower() == "laplace":
        return d.Laplace
    elif type_.lower() == "normal":
        return d.Normal
    elif type_.lower() == "kumaraswamy":
        return d.Kumaraswamy
    elif type_.lower() == "lognormal":
        return d.LogNormal
    else:
        try:
            getattr(d, type_)
        except AttributeError:
            raise NotImplementedError(f"unknown policy type {type_}")


def create_policy_nn(cfg, input_dim, output_dim):
    name = cfg["base"]["name"]
    if name.lower() in ("mlp", "fc"):
        return create_policy_mlp(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError(f"unknown neural network type {name}")


def create_policy_mlp(cfg, input_dim, output_dim):
    assert cfg["base"]["name"].lower() in ("mlp", "fc")

    policy_type = get_policy_type(cfg["type"])
    dist = get_dist_type(cfg["dist"])

    hidden = cfg["base"]["hidden"]
    act = cfg["base"]["activation"]
    bias = cfg["base"]["bias"]

    head_act = cfg["head_activation"]
    paths = policy_type(None, dist).n_params
    head_bias = cfg["head_bias"]

    head_layer_init = utils.init_layer(cfg["head_layer_init"])
    layer_init = utils.init_layer(cfg["base"]["layer_init"])

    assert len(hidden) == len(act)
    assert len(head_act) == paths

    net = []
    layer = nn.Linear(input_dim, hidden[0], bias=bias)
    layer = layer_init(layer)
    net.append(layer)
    net.append(init_activation(act[0]))

    for j in range(1, len(hidden)):
        layer = nn.Linear(hidden[j-1], hidden[j], bias=bias)
        layer = layer_init(layer)
        net.append(layer)
        net.append(init_activation(act[j]))

    head_layers = [[] for _ in range(paths)]
    for i in range(len(head_layers)):
        head_layer = nn.Linear(hidden[-1], output_dim, bias=head_bias)
        head_layer = head_layer_init(head_layer)
        head_layers[i].append(head_layer)

        for k in range(len(head_act[i])):
            head_layers[i].append(init_activation(head_act[i][k]))

    head = Parallel(*(nn.Sequential(*path) for path in head_layers))
    return nn.Sequential(nn.Sequential(*net), head)


def create_policy(cfg, input_dim, output_dim, action_min, action_max):
    net = create_policy_nn(cfg, input_dim, output_dim)

    policy_type = get_policy_type(cfg["type"])
    dist_type = get_dist_type(cfg["dist"])

    if policy_type is UnBounded:
        return policy_type(net, dist_type)
    else:
        return policy_type(
            net, dist_type, action_min=action_min, action_max=action_max,
        )


base_cfg = {
    "name": "fc",
    "hidden": [8, 8],
    "activation": [{"name": "relu"}, {"name": "relu"}],
    "bias": True,
    "layer_init": "xavier", # TODO
}

policy_cfg = {
    "base": base_cfg,
    "type": "bounded",
    "dist": "arctanhnormal",
    "head_activation": [
        [{"name": "none"}],
        [
            {"name": "tanh_shift", "args": (-4, 1, 50, 1), "kwargs": {}},
            {"name": "clamp", "args": (-20, 20), "kwargs": {}},
            {"name": "Exp", "args": tuple(), "kwargs": {}},
        ],
    ],
    "head_bias": True,
    "head_layer_init": "xavier",
}

p = create_policy(policy_cfg, 3, 2, 0, 1)
print(p)


# path1 = nn.Sequential(
#     nn.Linear(1, 10),
#     nn.ReLU(),
#     nn.Linear(10, 1),
#     nn.Softplus(1.0),
# )
# path2 = nn.Sequential(
#     nn.Linear(1, 13),
#     nn.ReLU(),
#     nn.Linear(13, 1),
#     nn.Softplus(0.5),
# )
# net = Parallel(path1, path2)
# policy = Bounded(net, ArctanhNormal, -2, 4)

# state = torch.zeros(5, 1)

# action, info = policy.forward(state)
# print()
# print("ACTIONS:", action)
# # print("INFO:", info)
# print()
# print(policy.log_prob(state, action))
