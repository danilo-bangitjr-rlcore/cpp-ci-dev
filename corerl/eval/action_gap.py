import torch
from corerl.agent.utils import get_top_action
from jaxtyping import Float

from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.utils.hydra import config


@config('action_gap', group='eval')
class ActionGapConfig(EvalConfig):
    name: str = 'action_gap'

    n_samples: int = 100
    offline_eval: bool = True
    online_eval: bool = True


class ActionGapEval(BaseEval):
    def __init__(self, cfg: ActionGapConfig, **kwargs):
        self.n_samples = cfg.n_samples
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'get_prev_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_prev_action_function'")
        if 'get_critic_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_critic_action_function'")
        self.get_prev_action = kwargs['get_prev_action_function']
        self.get_critic_action = kwargs['get_critic_action_function']

        self.agent = kwargs['agent']
        self.action_dim = self.agent.action_dim
        assert hasattr(self.agent, 'q_critic'), "Agent must have a q_critic"
        self.action_gaps: list[float] = []

    def _get_batch_action_gap(self, state_batch: Float[torch.Tensor, "batch_size state_dim"]) -> float:
        batch_size = state_batch.shape[0]

        states = torch.repeat_interleave(state_batch, self.n_samples, dim=0)
        # actor_actions = torch.rand(size=(batch_size * self.n_samples, self.action_dim))
        with torch.no_grad():
            actor_actions, _ = self.agent.actor.get_action(states)
        prev_action = self.get_prev_action(states)
        critic_actions = self.get_critic_action(actor_actions, prev_action)

        best_q_actions = get_top_action(
            self.agent.q_critic.get_q,
            states,
            critic_actions,
            self.action_dim,
            batch_size,
            n_actions=self.n_samples,
            return_idx=0,
        )
        best_pi_actions = get_top_action(self.agent.actor.get_log_prob, states, actor_actions,
                                         self.action_dim, batch_size, n_actions=self.n_samples, return_idx=0)

        best_q_actions = best_q_actions.squeeze()
        best_pi_actions = best_pi_actions.squeeze()

        # if actions are 1-dimensional, the previous squeeze would have reduced
        # best_q_actions, best_pi_actions down to shape (batch_size,)
        if len(best_q_actions.shape) == 1:
            action_gap = torch.norm(best_q_actions - best_pi_actions, p=2, dim=0)
        else:
            action_gap = torch.norm(best_q_actions - best_pi_actions, p=2, dim=1)

        mean_action_gap = torch.mean(action_gap)
        return mean_action_gap.item()

    def do_eval(self, **kwargs) -> None:
        batch = self.agent.critic_buffer.sample()
        batch = batch[0]
        state_batch = batch.state
        action_gap = self._get_batch_action_gap(state_batch)
        self.action_gaps.append(action_gap)

    def get_stats(self) -> dict:
        stats = {
            'action_gap': self.action_gaps
        }
        return stats
