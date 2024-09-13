from . import ContinuousPolicy
import torch
import torch.distributions as d


class Bounded(ContinuousPolicy):
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

    @classmethod
    @property
    def continuous(self):
        return True

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
