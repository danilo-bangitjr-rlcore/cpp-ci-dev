"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import torch
from omegaconf import DictConfig

from corerl.eval.base_eval import BaseEval
from corerl.data.data import TransitionBatch
from corerl.component.network.utils import to_np
from corerl.agent.utils import get_top_action
from jaxtyping import Float
from corerl.utils.device import device



class PolicyImprove(BaseEval):
    def __init__(self, cfg: DictConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'get_prev_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_prev_action_function'")
        if 'get_critic_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_critic_action_function'")

        self.agent = kwargs['agent']
        self.n_samples = cfg.n_samples
        self.get_prev_action = kwargs['get_prev_action_function']
        self.get_critic_action = kwargs['get_critic_action_function']
        self.policy_improvments: list[float] = []
        self.greedy_gaps: list[float] = []

    def _get_actions(self, state_batch: Float[torch.Tensor, "batch_size state_dim"],
                     prev_actions: Float[torch.Tensor, "batch_size action_dim"]) \
            -> [torch.Tensor, torch.Tensor]:
        batch_size = state_batch.shape[0]
        action_dim = prev_actions.shape[1]
        states = torch.repeat_interleave(state_batch, self.n_samples, dim=0)
        prev_action = self.get_prev_action(states)
        actor_actions = torch.rand(size=(batch_size * self.n_samples, action_dim))
        critic_actions = self.get_critic_action(actor_actions, prev_action)

        best_q_actions = get_top_action(
            self.agent.q_critic.get_q,
            states,
            critic_actions,
            action_dim,
            batch_size,
            n_actions=self.n_samples,
            return_idx=0
        )
        best_q_actions = best_q_actions.squeeze()
        best_q_actions = self.get_critic_action(best_q_actions, prev_actions)

        with torch.no_grad():
            sampled_actions, _ = self.agent.actor.get_action(states)
        best_pi_actions = get_top_action(
            self.agent.actor.get_log_prob,
            states,
            sampled_actions,
            action_dim,
            batch_size,
            n_actions=self.n_samples,
            return_idx=0
        )
        best_pi_actions = best_pi_actions.squeeze()
        best_pi_actions = self.get_critic_action(best_pi_actions, prev_actions)
        return best_q_actions, best_pi_actions

    def _estimate_improvement(self, batch: TransitionBatch) -> [list, list]:
        state_batch = batch.state
        dataset_action = batch.action
        prev_action_batch = self.get_prev_action(state_batch)

        greedy_action, learned_action = self._get_actions(state_batch, prev_action_batch)

        q_dataset, _ = self.agent.q_critic.get_qs([state_batch], [dataset_action], with_grad=False)
        q_learned, _ = self.agent.q_critic.get_qs([state_batch], [learned_action], with_grad=False)
        q_greedy, _ = self.agent.q_critic.get_qs([state_batch], [greedy_action], with_grad=False)

        policy_improvement = (q_learned - q_dataset).mean()
        greedy_gap = (q_greedy - q_learned).mean()
        return to_np(policy_improvement).tolist(), to_np(greedy_gap).tolist()

    def do_eval(self, **kwargs) -> None:
        batches = self.agent.critic_buffer.sample()
        batch = batches[0]
        policy_improvement, greedy_gap = self._estimate_improvement(batch)
        self.policy_improvments.append(policy_improvement)
        self.greedy_gaps.append(greedy_gap)

    def get_stats(self) -> dict:
        stats = {
            'policy_improvements': self.policy_improvments,
            'greedy_gaps': self.greedy_gaps
        }
        return stats
