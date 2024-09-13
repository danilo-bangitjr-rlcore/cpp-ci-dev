from . import ContinuousPolicy
import torch.distributions as d


class UnBounded(ContinuousPolicy):
    def __init__(self, model, dist):
        super().__init__(model)
        self._dist = dist

        if not isinstance(dist.support, type(d.constraints.real)):
            raise ValueError(
                "UnBounded expects dist to have support type " +
                f"{type(d.constraints.real)}, but " +
                f"got {type(dist.support)}"
            )

    @classmethod
    @property
    def continuous(self):
        return True

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
