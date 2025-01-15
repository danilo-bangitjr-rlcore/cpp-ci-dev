from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributions as d
from torch.types import _size
from typing_extensions import override

import corerl.utils.nullable as nullable


class ArctanhNormal(d.Distribution):
    _EPSILON = 1e-6

    has_rsample = True

    # Ignoring all these errors is probably not the best idea, but this is how
    # PyTorch implements things, so I am following their template
    support = d.constraints.interval(-1.0, 1.0)  # pyright: ignore [reportAttributeAccessIssue,reportIncompatibleMethodOverride]
    arg_constraints = { # pyright: ignore[reportIncompatibleMethodOverride,reportAssignmentType]
        "loc": d.constraints.real, "scale": d.constraints.positive,     # pyright: ignore [reportAttributeAccessIssue]
    }

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args: object | None = None):
        self._underlying = d.Normal(loc, scale, validate_args)

    @property
    def loc(self):
        """Get the location parameters
        """
        return self._underlying.loc

    @loc.setter
    def loc(self, value: float):
        """Set the location parameters
        """
        self._underlying.loc = value

    @property
    def scale(self):
        """Get the scale parameters
        """
        return self._underlying.scale

    @scale.setter
    def scale(self, value: float):
        """Set the scale parameters
        """
        self._underlying.scale = value

    @d.Distribution.batch_shape.getter
    @override
    def batch_shape(self):
        """Get the batch shape of the distribution
        """
        return self._underlying.batch_shape

    @d.Distribution.event_shape.getter
    @override
    def event_shape(self):
        """Get the event shape of the distribution
        """
        return self._underlying.event_shape

    @override
    def expand(self, batch_shape: Iterable[int], _instance: ArctanhNormal | None = None):
        new = self._get_checked_instance(ArctanhNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)

        assert isinstance(new, type(self))
        super(ArctanhNormal, new).__init__(batch_shape, validate_args=False)

        new._validate_args = self._validate_args
        return new

    @override
    def sample(self, sample_shape: _size | None = None):
        sample_shape = nullable.default(sample_shape, torch.Size)
        samples = self._underlying.sample(sample_shape=sample_shape)
        return torch.tanh(samples)

    @override
    def rsample(self, sample_shape: _size | None = None):
        sample_shape = nullable.default(sample_shape, torch.Size)
        samples = self._underlying.rsample(sample_shape=sample_shape)
        return torch.tanh(samples)

    @override
    def log_prob(self, value: torch.Tensor):
        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_logp = self._underlying.log_prob(normal_samples)

        offset = torch.log1p(
            -value.pow(2) + ArctanhNormal._EPSILON
        )

        return normal_logp - offset

    @override
    def cdf(self, value: torch.Tensor):
        less_than_mask = torch.where(value <= -1)
        greater_than_mask = torch.where(value >= 1)

        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_cdf = self._underlying.cdf(normal_samples)
        normal_cdf[less_than_mask] = 0
        normal_cdf[greater_than_mask] = 1

        return normal_cdf

    @override
    def icdf(self, value: torch.Tensor):
        less_than_mask = torch.where(value <= -1)
        greater_than_mask = torch.where(value >= 1)

        normal_samples = torch.atanh(torch.clamp(
            value, -1.0 + ArctanhNormal._EPSILON, 1.0 - ArctanhNormal._EPSILON,
        ))

        normal_cdf = self._underlying.icdf(normal_samples)
        normal_cdf[less_than_mask] = 1
        normal_cdf[greater_than_mask] = 0

        return normal_cdf

    @override
    def entropy(self):
        """
        Note that this function does not return the entropy of the
        ArctanhNormal distribution, which does not have a closed-form solution
        for the entropy function.

        Instead, this function enables the gradient of the distribution entropy
        to be correctly computed. A similar approach is taken in rlax.
        """
        return self._underlying.entropy()
