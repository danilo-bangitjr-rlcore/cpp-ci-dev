import torch
import torch.distributions as d

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

        assert isinstance(new, type(self))
        super(ArctanhNormal, new).__init__(batch_shape, validate_args=False)

        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape: torch.Size | None = None):
        sample_shape = nullable.default(sample_shape, torch.Size)
        samples = self._underlying.sample(sample_shape=sample_shape)
        return torch.tanh(samples)

    def rsample(self, sample_shape: torch.Size | None = None):
        sample_shape = nullable.default(sample_shape, torch.Size)
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
