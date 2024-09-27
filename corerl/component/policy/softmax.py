from . import Policy
import torch
import torch.nn as nn
import torch.distributions as d


class Softmax(Policy):
    def __init__(self, net, input_dim: int, output_dim: int):
        super(Softmax, self).__init__(net)
        self.output_dim = output_dim

    @classmethod
    def continuous(cls):
        return False

    def get_probs(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self._model(state)
        probs = nn.functional.softmax(x, dim=1)
        return probs, x

    @property
    def support(self):
        return d.constraints.integer_interval(0, self.output_dim-1)

    @property
    def param_names(self):
        return tuple(f"logit_{i}" for i in range(self.output_dim))

    @classmethod
    def from_env(cls, model, env):
        output_dim = env.action_space.shape[0]
        input_dim = env.observation_space.shape[0]

        return cls(model, input_dim, output_dim)

    def forward(
            self, state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        probs, x = self.get_probs(state)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()

        logp = dist.log_prob(actions)
        logp = logp.view((logp.shape[0], 1))

        actions = actions.reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.output_dim)
        a_onehot.zero_()
        actions = a_onehot.scatter_(1, actions, 1)
        return actions, {'logp': logp}

    def log_prob(
            self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        actions = (actions == 1).nonzero(as_tuple=False)
        actions = actions[:, 1:]
        probs, _ = self.get_probs(states)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions.squeeze(-1))
        logp = logp.view(-1, 1)
        return logp, {}
