"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import numpy as np
import torch
from omegaconf import DictConfig

from corerl.eval.base_eval import BaseEval
from corerl.component.network.utils import to_np
from corerl.agent.utils import get_top_action


class Curvature(BaseEval):
    def __init__(self, cfg: DictConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'get_prev_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_prev_action_function'")
        if 'get_critic_action_function' not in kwargs:
            raise KeyError("Missing required argument: 'get_critic_action_function'")

        self.agent = kwargs['agent']
        self.get_prev_action = kwargs['get_prev_action_function']
        self.get_critic_action = kwargs['get_critic_action_function']

        self.n_samples = cfg.num_action_samples
        self.num_deltas = cfg.num_deltas
        self.epsilon = cfg.epsilon
        self.relaxation = cfg.relaxation
        r = torch.rand((1, self.num_deltas, self.agent.action_dim))
        self.delta_action = (1 - self.epsilon) * r + self.epsilon

        self.local_max = []
        self.local_min = []
        self.linear = []
        self.flat = []

    def do_eval(self, **kwargs) -> None:
        # estimate the td error on a batch
        batches = self.agent.critic_buffer.sample()
        batch = batches[0]
        action_dim = self.agent.action_dim
        state_batch = batch.state
        prev_action_batch = self.get_prev_action(state_batch)
        batch_size = state_batch.size(0)

        # get best action selected by the actor network
        states = torch.repeat_interleave(state_batch, self.n_samples, dim=0)
        with torch.no_grad():
            actions, _ = self.agent.actor.get_action(states)
        best_pi_actions = get_top_action(
            self.agent.actor.get_log_prob,
            states,
            actions,
            action_dim,
            batch_size,
            n_actions=self.n_samples,
            return_idx=0
        )
        best_pi_actions = best_pi_actions.squeeze()

        repeated_states = state_batch.repeat_interleave(self.num_deltas, dim=0)
        repeated_action_batch = best_pi_actions.repeat_interleave(self.num_deltas, dim=0).view(
            (batch_size, self.num_deltas, -1)
        )
        repeated_prev_action = self.get_prev_action(repeated_states)

        theta_plus = (repeated_action_batch + self.delta_action).view(
            (repeated_states.size(0), -1)
        ).clip(0, 1)
        theta_minus = (repeated_action_batch - self.delta_action).view(
            (repeated_states.size(0), -1)
        ).clip(0, 1)

        best_pi_actions = self.get_critic_action(best_pi_actions, prev_action_batch)
        theta_plus = self.get_critic_action(theta_plus, repeated_prev_action)
        theta_minus = self.get_critic_action(theta_minus, repeated_prev_action)

        q_0, _ = self.agent.q_critic.get_qs([state_batch], [best_pi_actions], with_grad=False)
        q_plus, _ = self.agent.q_critic.get_qs([repeated_states], [theta_plus], with_grad=False)
        q_minus, _ = self.agent.q_critic.get_qs([repeated_states], [theta_minus], with_grad=False)

        q_0 = q_0.repeat_interleave(self.num_deltas, dim=1)
        q_plus = q_plus.view((batch_size, self.num_deltas))
        q_minus = q_minus.view((batch_size, self.num_deltas))
        delta_plus = to_np(q_plus - q_0)
        delta_minus = to_np(q_minus - q_0)
        relaxation = to_np(q_0.max() - q_0.min()) * self.relaxation

        less_less = ((delta_plus <= relaxation) & (delta_minus <= relaxation)).astype(
            int).sum(axis=1) / float(self.num_deltas)
        larger_larger = ((delta_plus > relaxation) & (delta_minus > relaxation)).astype(
            int).sum(axis=1) / float(self.num_deltas)
        equal = np.isclose(delta_plus, -delta_minus).astype(
            int).sum(axis=1) / float(self.num_deltas)
        zero = (np.isclose(delta_plus, 0.) & np.isclose(delta_minus, 0.)).astype(
            int).sum(axis=1) / float(self.num_deltas)

        self.local_max.append(less_less.mean())
        self.local_min.append(larger_larger.mean())
        self.linear.append(equal.mean())
        self.flat.append(zero.mean())

    def get_stats(self) -> dict:
        stats = {
            'local_max': self.local_max,
            'local_min': self.local_min,
            'linear': self.linear,
            'flat': self.flat,
        }
        return stats
