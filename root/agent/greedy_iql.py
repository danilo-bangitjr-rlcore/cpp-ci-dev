from omegaconf import DictConfig
import torch

from root.agent.iql import IQL

class GreedyIQL(IQL):
    """
    A verison of IQL that uses GAC-style updates. We assume use of a uniform-proposal policy
    """
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.num_samples = cfg.num_samples
        self.top_action = cfg.top_actions

    def compute_actor_loss(self, batch: dict) -> torch.Tensor:
        state_batch = batch["states"]
        repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions = self.get_policy_update_data(state_batch)
        logp, _ = self.actor.get_log_prob(stacked_s_batch, best_actions, with_grad=True)
        actor_loss = -logp.mean()
        return actor_loss

    def sort_q_value(self, repeated_states: torch.Tensor, sample_actions: torch.Tensor, batch_size: int) -> torch.Tensor:
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py
        q_values, _ = self.q_critic.get_q(repeated_states, sample_actions, with_grad=False)
        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q

    # Assuming uniform proposal policy
    def get_policy_update_data(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        sample_actions = torch.rand(self.num_samples)
        sample_actions = sample_actions.repeat(batch_size).reshape(-1, 1)

        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_action]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.action_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions
