import torch
import numpy as np
from src.agent.greedy_ac import GreedyAC


class CQLGAC(GreedyAC):

    def __init__(self, cfg):
        super(CQLGAC, self).__init__(cfg)
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        if cfg.discrete_control:
            self.top_action = 1

    def critic_loss(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        q_value, _ = self.get_q_value(state_batch, action_batch, with_grad=True)
        q_loss_main = torch.nn.functional.mse_loss(target, q_value)

        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)
        repeated_next_states = next_state_batch.repeat_interleave(self.num_samples, dim=0)

        cql_random_actions = action_batch.new_empty((batch_size * self.num_samples, action_dim),
                                                    requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis, _ = self.get_policy(repeated_states, with_grad=False)
        cql_next_actions, cql_next_log_pis, _ = self.get_policy(repeated_next_states, with_grad=False)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

        cql_q1_rand, _ = self.get_q_value(repeated_states, cql_random_actions, with_grad=True)
        cql_q1_current_actions, _ = self.get_q_value(repeated_states, cql_current_actions, with_grad=True)
        cql_q1_next_actions, _ = self.get_q_value(repeated_states, cql_next_actions, with_grad=True)

        # resize to [batch, sample, 1]
        cql_q1_rand = cql_q1_rand.reshape(batch_size, self.num_samples, 1)
        cql_q1_next_actions = cql_q1_next_actions.reshape(batch_size, self.num_samples, 1)
        cql_q1_current_actions = cql_q1_current_actions.reshape(batch_size, self.num_samples, 1)
        print("next q size", next_q.size(), cql_q1_rand.size(), cql_q1_current_actions.size(), cql_q1_next_actions.size())

        # importance sampling
        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1 = [
            torch.cat(
                [qrand - random_density,
                 qna - cql_next_log_pis.detach(),
                 qca - cql_current_log_pis.detach()],
                dim=1
            ) for qrand, qna, qca in zip(cql_q1_rand, cql_q1_next_actions, cql_q1_current_actions)
        ]

        cql_qf1_ood = [
            torch.logsumexp(catq / self.cql_temp, dim=1) * self.cql_temp
            for catq in cql_cat_q1
        ]

        print("qpred size", q_value, cql_qf1_ood)
        exit()
        """Subtract the log likelihood of data"""
        cql_qf1_diff = [
            torch.clamp(
                qood - qpred,
                self.cql_clip_diff_min,
                self.cql_clip_diff_max,
            ).mean()
            for qood, qpred in zip(cql_qf1_ood, q_value)
        ]

        cql_min_qf1_loss = [
            qdiff * self.cql_min_q_weight
            for qdiff in cql_qf1_diff
        ]

        q_loss = q_loss_main + cql_min_qf1_loss.mean()
        return q_loss, next_action