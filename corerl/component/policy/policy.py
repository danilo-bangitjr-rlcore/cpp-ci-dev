from abc import ABC, abstractmethod, abstractclassmethod
import torch
import torch.distributions as d


class Policy(ABC):
    def __init__(self, model):
        self._model = model

    def load_state_dict(self, sd):
        return self._model.load_state_dict(sd)

    def state_dict(self):
        return self._model.state_dict()

    def parameters(self):
        return self._model.parameters()

    @classmethod
    @abstractmethod
    def from_env(cls, model, dist, env):
        pass

    @property
    @abstractmethod
    def param_names(self):
        pass

    @property
    def n_params(self):
        return len(self.param_names)

    @property
    @abstractclassmethod
    def continuous(cls):
        pass

    @classmethod
    @property
    def discrete(cls):
        return not cls.continuous

    @property
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def forward(self, state, rsample=True):
        pass

    @abstractmethod
    def log_prob(self, state: torch.Tensor, action: torch.Tensor):
        pass


class ContinuousPolicy(Policy,ABC):
    def __init__(self, model):
        super().__init__(model)

    def load_state_dict(self, sd):
        return self._model.load_state_dict(sd)

    def state_dict(self):
        return self._model.state_dict()

    def parameters(self):
        return self._model.parameters()

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
    @abstractclassmethod
    def continuous(cls):
        pass

    @classmethod
    @property
    def discrete(cls):
        return not cls.continuous

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

    def sample(self, state):
        return self.forward(state, False)

    def rsample(self, state):
        return self.forward(state, True)

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
