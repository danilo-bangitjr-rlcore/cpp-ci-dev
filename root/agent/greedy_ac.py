from omegaconf import DictConfig
from pathlib import Path

import torch.nn as nn
import torch
import numpy
import pickle as pkl

from root.agent.base import BaseAC
from root.component.actor.factory import init_actor
from root.component.critic.factory import init_q_critic
from root.component.buffer.factory import init_buffer
from root.component.network.utils import to_np, state_to_tensor


class GreedyAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.device = cfg.device
        self.action_dim = action_dim
        # Removed self.gac_a_dim = self.action_dim. Hopefully this doesn't break anything

        self.average_entropy = cfg.average_entropy  # Whether to average the proposal policy's entropy over all the sampled actions
        self.tau = cfg.tau  # Entropy constant used in the entropy version of the proposal policy update
        self.rho = cfg.rho  # percentage of sampled actions used in actor update
        self.rho_proposal = self.rho * cfg.prop_rho_mult  # percentage of sampled actions used in the non-entropy version of the proposal policy update
        self.num_samples = cfg.num_samples  # number of actions sampled from the proposal policy
        self.top_actions = int(self.rho * self.num_samples)  # Number of actions used to update actor
        self.top_actions_proposal = int(
            self.rho_proposal * self.num_samples)  # Number of actions used to update proposal policy

        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.sampler = init_actor(cfg.actor, state_dim, action_dim, initializer=self.actor)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

        if self.discrete_control:
            self.top_actions = 1
            if self.num_samples > self.action_dim:
                self.get_policy_update_data = self.get_policy_update_data_discrete
                self.num_samples = self.action_dim
                self.top_actions_proposal = self.action_dim

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, self.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]

        return action

    def update_buffer(self, transition: tuple) -> None:
        self.buffer.feed(transition)

    def sort_q_value(self, repeated_states: torch.Tensor, sample_actions: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py

        q_values = self.q_critic.get_q(repeated_states, sample_actions, with_grad=False)
        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)

        return sorted_q

    def get_policy_update_data(self, state_batch: torch.Tensor) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)
        sample_actions, _ = self.sampler.get_action(repeated_states, with_grad=False)

        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_actions]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.action_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def get_policy_update_data_discrete(self, state_batch: torch.Tensor) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.action_dim, dim=0)
        actions = torch.arange(self.action_dim).reshape((1, -1))
        actions = actions.repeat_interleave(batch_size, dim=0).reshape((-1, 1))
        a_onehot = torch.FloatTensor(actions.size()[0], self.action_dim)
        a_onehot.zero_()
        sample_actions = a_onehot.scatter_(1, actions, 1)

        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_actions]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.action_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def compute_critic_loss(self, batch: dict) -> (list, torch.Tensor):
        state_batch = batch['states']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        next_state_batch = batch['next_states']
        mask_batch = 1 - batch['dones']

        next_actions, _ = self.actor.get_action(next_state_batch, with_grad=False)
        next_q = self.q_critic.get_q_target(next_state_batch, next_actions)
        target = reward_batch + mask_batch * self.gamma * next_q
        q = self.q_critic.get_q(state_batch, action_batch, with_grad=True)
        q_loss = nn.functional.mse_loss(q, target)

        return q_loss

    def compute_sampler_entropy_loss(self, extra_info: tuple, batch_size: int) -> torch.Tensor:
        repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions = extra_info
        sample_actions = sample_actions.reshape(-1, self.action_dim)
        sampler_entropy, _ = self.sampler.get_log_prob(repeated_states, sample_actions)

        with torch.no_grad():
            sampler_entropy *= sampler_entropy
        sampler_entropy = sampler_entropy.reshape(batch_size, self.num_samples, 1)

        if self.average_entropy:
            sampler_entropy = -sampler_entropy.mean(axis=1)
        else:
            sampler_entropy = -sampler_entropy[:, 0, :]

        logp, _ = self.sampler.get_log_prob(stacked_s_batch, best_actions, with_grad=True)
        sampler_loss = logp.reshape(batch_size, self.top_actions, 1)
        sampler_loss = -1 * (sampler_loss.mean(axis=1) + self.tau * sampler_entropy).mean()

        return sampler_loss

    def compute_sampler_no_entropy_loss(self, extra_info: tuple, batch_size: int, states: torch.Tensor) -> torch.Tensor:
        (repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions) = extra_info
        best_ind_proposal = sorted_q[:, :self.top_actions_proposal]
        best_ind_proposal = best_ind_proposal.repeat_interleave(self.action_dim, -1)
        best_actions_proposal = torch.gather(sample_actions, 1, best_ind_proposal)
        stacked_s_batch_proposal = states.repeat_interleave(self.top_actions_proposal, dim=0)
        best_actions_proposal = torch.reshape(best_actions_proposal, (-1, self.action_dim))

        logp, _ = self.sampler.get_log_prob(stacked_s_batch_proposal, best_actions_proposal, with_grad=True)
        sampler_loss = logp.reshape(batch_size, self.top_actions_proposal, 1)
        sampler_loss = -1 * (sampler_loss.mean(axis=1)).mean()

        return sampler_loss

    def compute_actor_loss(self, data) -> (torch.Tensor, tuple):
        states = data['states']
        repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions = self.get_policy_update_data(states)
        logp, _ = self.actor.get_log_prob(stacked_s_batch, best_actions, with_grad=True)
        actor_loss = -logp.mean()

        return actor_loss, (repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions)

    def compute_actor_sampler_losses(self, data) -> (torch.Tensor, torch.Tensor):
        actor_loss, extra_info = self.compute_actor_loss(data)
        states = data['states']
        batch_size = states.shape[0]
        if self.tau != 0:  # Entropy version of proposal policy update
            sampler_loss = self.compute_sampler_entropy_loss(extra_info, batch_size)
        else:
            # Non-entropy version of proposal policy update.
            # A greater percentage of actions are used to update the proposal policy than the actor policy
            sampler_loss = self.compute_sampler_no_entropy_loss(extra_info, batch_size, states)

        return actor_loss, sampler_loss

    def update_critic(self) -> None:
        batch = self.buffer.sample()
        q_loss = self.compute_critic_loss(batch)
        self.q_critic.update(q_loss)

    def update_actor(self) -> None:
        batch = self.buffer.sample()
        actor_loss, sampler_loss = self.compute_actor_sampler_losses(batch)
        self.actor.update(actor_loss)
        self.sampler.update(sampler_loss)

    def update(self) -> None:
        self.update_critic()
        self.update_actor()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        sampler_path = path / "sampler"
        self.sampler.save(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        sampler_path = path / "sampler"
        self.sampler.load(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)


class GreedyACUniformProp(GreedyAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.top_action = cfg.top_actions

    def get_policy_update_data(self, state_batch: torch.Tensor) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        batch_size = state_batch.shape[0]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        # Sample actions from uniform proposal policy
        sample_actions = torch.rand((self.num_samples, self.action_dim))
        sample_actions = sample_actions.repeat_interleave(batch_size, dim=0)
        sorted_q = self.sort_q_value(repeated_states, sample_actions, batch_size)
        best_ind = sorted_q[:, :self.top_actions]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)

        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.action_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        return repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions

    def update_actor(self) -> None:
        batch = self.buffer.sample()
        pi_loss, _ = self.compute_actor_loss(batch)  # Update Actor. No need to update proposal policy
        self.actor.update(pi_loss)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)


class GreedyACLineSearch(GreedyAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.actor.set_parameters(id(self.buffer))
        self.sampler.set_parameters(id(self.buffer))
        self.q_critic.set_parameters(id(self.buffer), eval_error_fn=self.critic_eval_error_fn)

    def critic_eval_error_fn(self, args):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = args
        q = self.q_critic.get_q(state_batch, action_batch, with_grad=False)
        next_action, _ = self.actor.get_action(next_state_batch, with_grad=False)
        next_q = self.q_critic.get_q_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        error = torch.nn.functional.mse_loss(q.detach(), target.detach())
        return error
