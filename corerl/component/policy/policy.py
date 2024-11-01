from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.distributions as d
import torch.distributions.constraints as constraints
import logging
from typing import Union, Optional, Any, Mapping, Iterator
from typing_extensions import override

import corerl.utils.nullable as nullable

_BoundedAboveConstraint = constraints.less_than

_BoundedBelowConstraint = Union[
    constraints.greater_than_eq,
    constraints.greater_than,
]

_HalfBoundedConstraint = Union[
    _BoundedAboveConstraint,
    _BoundedBelowConstraint,
]


class Policy(ABC):
    """
    Policy is the abstract base class for all policies

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch module to use as the policy's neural network
    """
    def __init__(self, model: nn.Module):
        self._model = model

    @property
    @abstractmethod
    def has_rsample(self) -> bool:
        raise NotImplementedError()

    def load_state_dict(self, sd: Mapping[str, Any]):
        """
        Loads the state dictionary `sd` into the policy's model
        """
        return self._model.load_state_dict(sd)

    def state_dict(self) -> Mapping[str, Any]:
        """
        Gets the state dictionary from the policy's model
        """
        return self._model.state_dict()

    def parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns the parameters of the policy's model.
        """
        return self._model.parameters()

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """
        The names of the policy distribution parameters
        """
        pass

    @property
    def n_params(self) -> int:
        """
        The number of parameters that the policy distribution has
        """
        return len(self.param_names)

    @classmethod
    @abstractmethod
    def continuous(cls) -> bool:
        """
        Whether the policy class represents a continuous-action policy or not
        """
        pass

    @classmethod
    def discrete(cls) -> bool:
        """
        Whether the policy class represents a discrete-action policy or not
        """
        return not cls.continuous

    @property
    @abstractmethod
    def support(self) -> constraints.Constraint:
        """
        The support of the policy distribution
        """
        pass

    @abstractmethod
    def forward(self, state, rsample=True) -> tuple[torch.Tensor, dict]:
        """
        Return a sample from the policy in state `state`
        """
        pass

    @abstractmethod
    def log_prob(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Return the log density/probability of the action `action` in the state
        `state`
        """
        pass


class ContinuousIIDPolicy(Policy,ABC):
    """
    ContinuousIIDPolicy represents a continuous-action policy, where each
    action dimension is IID.
    """
    def __init__(self, model, dist):
        super().__init__(model)
        self._dist = dist

    @classmethod
    def from_(cls, model, dist, *args, **kwargs):
        """
        Factory which returns a ContinuousIIDPolicy which supports the
        distribution `dist`.

        `args` and `kwargs` are passed to the constructor of the concrete
        ContinuousIIDPolicy type:

        - If `dist` has bounded support, then you should pass 2 (keyword)
        arguments, corresponding to (`action_min`, `action_max`)
        - If `dist` is bounded above and unbounded below, then you should pass
        1 (keyword) argument, corresponding to (`action_max`,)
        - If `dist` is bounded below and unbounded above, then you should pass
        1 (keyword) argument, corresponding to (`action_min`,)
        - If `dist` is unbounded, then no extra arguments should be passed

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch module to use as the policy's neural network
        dist : torch.distribution.Distribution
            The policy distribution to use
        *args : Iterable[any]
            The arguments to pass to the policy constructor
        **kwargs : Dict[any]
            The keyword arguments to pass to the policy constructor

        Returns
        -------
        ContinuousIIDPolicy
        """
        return _get_type_from_dist(dist)(model, dist, *args, **kwargs)

    @classmethod
    @abstractmethod
    def from_env(cls, model, dist, env) -> 'ContinuousIIDPolicy':
        pass

    def _transform_from_params(self, *params) -> d.Distribution:
        """
        Given a tuple of policy parameters, return the underlying policy
        distribution transformed to cover the correct support region.

        For example, if the underlying distribution is a Beta distribution and
        the support region is (10, 11), then this function takes a 2-tuple of
        parameters for the Beta distribution and returns a Beta distribution
        with support over (10, 11)

        `params` must satisfy `params[i].shape == params[j].shape ∀i,j`.
        """
        return self._transform(self._dist(*params))

    @abstractmethod
    def _transform(self, dist) -> d.Distribution:
        """
        Similar to _transform_from_params, but takes the underlying
        distribution object to transform, rather than its parameters.
        """
        pass

    @Policy.param_names.getter
    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        pass

    @Policy.n_params.getter
    def n_params(self):
        return len(self.param_names)

    @override
    @classmethod
    def continuous(cls):
        return True

    @Policy.support.getter
    @abstractmethod
    def support(self) -> constraints.Constraint:
        pass

    @override
    def forward(
        self,
        state: torch.Tensor,
        rsample: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        params = self._model(state)
        dist = self._transform_from_params(*params)

        info = dict(zip(
            [param_name for param_name in self.param_names],
            [p.squeeze().detach().numpy() for p in params],
            strict=True,
        ))

        if rsample:
            samples = dist.rsample()
        else:
            samples = dist.sample()

        return samples, info

    def sample(self, state) -> tuple[torch.Tensor, dict]:
        return self.forward(state, False)

    def rsample(self, state) -> tuple[torch.Tensor, dict]:
        return self.forward(state, True)

    @override
    def log_prob(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        params = self._model(state)
        dist = self._transform_from_params(*params)

        assert dist.support is not None
        if not torch.all(dist.support.check(action)):
            raise ValueError(
                "expected all actions to be within the distribution support " +
                f"of {dist.support}, but got actions: \n{action}"
            )

        lp = dist.log_prob(action)
        lp = lp.view(-1, 1)

        return lp, {}

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.mean

    def mode(self, state: torch.Tensor) -> torch.Tensor:
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.mode

    def variance(self, state: torch.Tensor) -> torch.Tensor:
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.variance

    def stddev(self, state: torch.Tensor) -> torch.Tensor:
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.stddev

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        params = self._model(state)
        dist = self._transform_from_params(*params)
        return dist.entropy()

    def kl(self, other, state) -> torch.Tensor:
        self_params = self._model(state)
        self_dist = self._transform_from_params(*self_params)

        other_params = other._model(state)
        other_dist = other._transform_from_params(*other_params)

        return d.kl.kl_divergence(self_dist, other_dist)

    @Policy.has_rsample.getter
    def has_rsample(self) -> bool:
        return self._dist.has_rsample


class Bounded(ContinuousIIDPolicy):
    def __init__(
        self,
        model: nn.Module,
        dist: type[d.Distribution],
        action_min: torch.Tensor | float | None,
        action_max: torch.Tensor | float | None,
    ):
        super().__init__(model, dist)

        if not isinstance(dist.support, constraints.interval):
            raise ValueError(
                "Bounded expects dist to have support type " +
                f"{constraints.interval}, but " +
                f"got {type(dist.support)}"
            )
        dist_min = dist.support.lower_bound
        dist_max = dist.support.upper_bound

        action_min = nullable.default(
            action_min,
            lambda: torch.Tensor([self._dist.support.lower_bound]),
        )
        action_min = torch.as_tensor(action_min)

        action_max = nullable.default(
            action_max,
            lambda: torch.Tensor([self._dist.support.upper_bound]),
        )
        action_max = torch.as_tensor(action_max)

        assert torch.all(action_min < action_max)

        self._action_scale = (action_max - action_min) / (dist_max - dist_min)
        self._action_bias = -self._action_scale * dist_min + action_min

    @ContinuousIIDPolicy.support.getter
    def support(self) -> constraints.Constraint:
        lb = self._dist.support.lower_bound
        ub = self._dist.support.upper_bound

        return type(self._dist.support)(
            lower_bound=self._action_scale * lb + self._action_bias,
            upper_bound=self._action_scale * ub + self._action_bias,
        )

    @ContinuousIIDPolicy.param_names.getter
    def param_names(self) -> tuple[str, ...]:
        return tuple(self._dist.arg_constraints.keys())

    @override
    @classmethod
    def from_env(
        cls,
        model: nn.Module,
        dist: type[d.Distribution],
        env: Any,
    ) -> 'Bounded':
        action_min = torch.Tensor(env.action_space.low)
        action_max = torch.Tensor(env.action_space.high)

        return cls(model, dist, action_min, action_max)

    @override
    def _transform(self, dist: d.Distribution) -> d.Distribution:
        if self._action_bias != 0 or self._action_scale != 1:
            transform = d.AffineTransform(
                loc=self._action_bias, scale=self._action_scale,
            )
            dist = d.TransformedDistribution(dist, [transform])

        return d.Independent(dist, 1)


class UnBounded(ContinuousIIDPolicy):
    def __init__(self, model: nn.Module, dist: type[d.Distribution]):
        super().__init__(model, dist)

        if not isinstance(dist.support, type(constraints.real)):
            raise ValueError(
                "UnBounded expects dist to have support type " +
                f"{type(constraints.real)}, but " +
                f"got {type(dist.support)}"
            )

    @ContinuousIIDPolicy.support.getter
    def support(self) -> constraints.Constraint:
        return self._dist.support

    @ContinuousIIDPolicy.param_names.getter
    def param_names(self) -> tuple[str, ...]:
        return tuple(self._dist.arg_constraints.keys())

    @override
    @classmethod
    def from_env(
        cls,
        model: nn.Module,
        dist: type[d.Distribution],
        env: Any,
    ) -> 'UnBounded':
        return cls(model, dist)

    @override
    def _transform(self, dist: d.Distribution) -> d.Distribution:
        return d.Independent(dist, 1)


class HalfBounded(ContinuousIIDPolicy):
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
    def __init__(
        self,
        model: nn.Module,
        dist: type[d.Distribution],
        action_min: Optional[torch.Tensor]=None,
        action_max: Optional[torch.Tensor]=None,
    ):
        super().__init__(model, dist)

        if isinstance(dist.support, _BoundedAboveConstraint):
            self._dist_max = dist.support.upper_bound
            self._dist_min = -torch.inf
            if action_min is not None and torch.all(action_min != -torch.inf):
                logging.warning(
                    f"the support of {dist} is not bounded below, ignoring" +
                    "action_min value {action_min}"
                )
            action_min = torch.tensor(-torch.inf)

            if action_max is None:
                action_max = torch.Tensor([self._dist.support.upper_bound])

            assert isinstance(action_max, torch.Tensor)
            assert torch.all(action_max != torch.inf)

        elif isinstance(dist.support, _BoundedBelowConstraint):
            self._dist_min = dist.support.lower_bound
            self._dist_max = torch.inf
            if action_max is not None and torch.all(action_max != torch.inf):
                logging.warning(
                    f"the support of {dist} is not bounded above, ignoring " +
                    f"action_max value {action_max}"
                )
            action_max = torch.tensor(torch.inf)

            if action_min is None:
                action_min = torch.Tensor([self._dist.support.lower_bound])

            assert isinstance(action_min, torch.Tensor)
            assert torch.all(action_min != -torch.inf)

        else:
            raise ValueError(
                "HalfBounded expects dist to have support type " +
                f"{_HalfBoundedConstraint}, but " +
                f"got {type(dist.support)}"
            )

        assert action_max is not None and action_min is not None
        assert torch.all(action_min < action_max)
        self._action_min = action_min
        self._action_max = action_max

    @ContinuousIIDPolicy.support.getter
    def support(self) -> constraints.Constraint:
        if self._bounded_below:
            lb = self._dist.support.lower_bound
            return type(self._dist.support)(lb + self._action_min)
        else:
            ub = self._dist.support.upper_bound
            return type(self._dist.support)(ub + self._action_max)

    @property
    def _bounded_below(self) -> bool:
        return self._dist_min is not -torch.inf

    @property
    def _bounded_above(self) -> bool:
        return self._dist_max is not torch.inf

    @ContinuousIIDPolicy.param_names.getter
    def param_names(self) -> tuple[str, ...]:
        return tuple(self._dist.arg_constraints.keys())

    @override
    @classmethod
    def from_env(
        cls,
        model: nn.Module,
        dist: type[d.Distribution],
        env: Any,
    ) -> 'HalfBounded':
        action_min = torch.Tensor(env.action_space.low)
        action_max = torch.Tensor(env.action_space.high)

        return cls(model, dist, action_min, action_max)

    @override
    def _transform(self, dist: d.Distribution) -> d.Distribution:
        if self._bounded_below:
            transform = d.AffineTransform(loc=self._action_min, scale=1)
            dist = d.TransformedDistribution(dist, [transform])
        else:
            transform = d.AffineTransform(loc=self._action_max, scale=1)

        dist = d.TransformedDistribution(dist, [transform])

        return d.Independent(dist, 1)

    def __repr__(self) -> str:
        return f"HalfBounded[{self._dist}, {self._model}]"


def _get_type_from_dist(
    dist: d.Distribution | type[d.Distribution],
) -> type[ContinuousIIDPolicy]:
    if isinstance(dist.support, constraints.interval):
        return Bounded

    # Weirdness with PyTorch's constraints.real makes us need to call `type` on
    # it first
    elif isinstance(dist.support, type(constraints.real)):
        return UnBounded

    elif isinstance(dist.support, _HalfBoundedConstraint):
        return HalfBounded

    raise NotImplementedError(
        f"unknown policy type for distribution {dist}",
    )
