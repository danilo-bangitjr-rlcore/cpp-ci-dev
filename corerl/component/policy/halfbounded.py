from typing import Union
import torch
import torch.distributions as d
from . import Policy


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
