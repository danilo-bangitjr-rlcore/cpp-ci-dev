from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy
import pickle as pkl

from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor, ensemble_mse
from corerl.component.exploration.factory import init_exploration_module
from corerl.utils.device import device
from corerl.data import TransitionBatch, Transition
import corerl.agent.utils as utils
import corerl.utils.freezer as fr

from jaxtyping import Float
from typing import Optional


class GreedyAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.ensemble_targets = cfg.ensemble_targets

        self.action_dim = action_dim
        # Removed self.gac_a_dim = self.action_dim. Hopefully this doesn't break anything

        self.average_entropy = cfg.average_entropy  # Whether to average the proposal policy's entropy over all the sampled actions
        self.tau = cfg.tau  # Entropy constant used in the entropy version of the proposal policy update
        self.rho = cfg.rho  # percentage of sampled actions used in actor update
        self.rho_proposal = self.rho * cfg.prop_rho_mult  # percentage of sampled actions used in the non-entropy version of the proposal policy update

        self.num_samples = cfg.num_samples  # number of actions sampled from the proposal policy
        self.share_batch = cfg.share_batch  # whether updates to proposal and actor should share a batch
        self.uniform_proposal = cfg.uniform_proposal  # whether to use a uniform proposal policy

        self.n_sampler_updates = cfg.n_sampler_updates
        if self.share_batch:
            assert self.n_critic_updates == self.n_sampler_updates, "Actor and proposal must use same number of updates"

        self.top_actions = int(self.rho * self.num_samples)  # Number of actions used to update actor
        self.top_actions_proposal = int(
            self.rho_proposal * self.num_samples)  # Number of actions used to update proposal policy

        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.sampler = init_actor(cfg.actor, state_dim, action_dim, initializer=self.actor)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

        self.sample_all_discrete_actions = False
        if self.discrete_control:
            self.top_actions = 1
            if self.num_samples >= self.action_dim:
                self.num_samples = self.action_dim
                self.top_actions_proposal = self.action_dim

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device)
        tensor_action, action_info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        # log the action_info to the freezer
        fr.freezer.store('action_info', action_info)
        return action

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def sort_q_value(self, repeated_states: torch.Tensor, sample_actions: torch.Tensor,
                     batch_size: int) -> Float[torch.Tensor, 'batch_size num_samples']:
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py

        q_values: Float[torch.Tensor, 'batch_size*num_samples 1']
        q_values = self.q_critic.get_q(repeated_states, sample_actions, with_grad=False)
        q_values: Float[torch.Tensor, 'batch_size num_samples 1']
        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q_inds: Float[torch.Tensor, 'batch_size num_samples 1']
        sorted_q_inds = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q_inds

    def get_policy_update_info(self, state_batch: Float[torch.Tensor, 'batch_size state_dim']) -> (
            Float[torch.Tensor, 'batch_size state_dim'],
            Float[torch.Tensor, 'batch_size*num_samples state_dim'],
            Float[torch.Tensor, 'batch_size num_samples action_dim'],
            Float[torch.Tensor, 'batch_size num_samples'],
            Float[torch.Tensor, 'batch_size*top_actions state_dim'],
            Float[torch.Tensor, 'batch_size*num_samples action_dim'],
            int):

        batch_size = state_batch.shape[0]
        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        repeated_states: Float[torch.Tensor, 'batch_size*num_samples state_dim']
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        if self.sample_all_discrete_actions:  # if discrete control AND we are sampling all actions
            sample_actions: Float[torch.Tensor, 'batch_size*action_dim action_dim']
            sample_actions = utils.get_batch_actions_discrete(state_batch, self.action_dim)
        else:  # o/w sample from sampler
            if self.uniform_proposal:
                # for the continuous case
                if self.discrete_control:
                    sample_actions = utils.get_batch_actions_discrete(state_batch, self.action_dim,
                                                                      samples=self.num_samples)
                else:
                    sample_actions = torch.rand((self.num_samples * batch_size, self.action_dim))
            else:
                sample_actions: Float[torch.Tensor, 'batch_size*action_dim action_dim']
                sample_actions, _ = self.sampler.get_action(repeated_states, with_grad=False)

        sorted_q_inds: Float[torch.Tensor, 'batch_size num_samples']
        sorted_q_inds = self.sort_q_value(repeated_states, sample_actions, batch_size)

        best_ind: Float[torch.Tensor, 'batch_size top_actions 1']
        best_ind = sorted_q_inds[:, :self.top_actions]

        best_ind: Float[torch.Tensor, 'batch_size top_actions action_dim']
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)

        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        sample_actions: Float[torch.Tensor, 'batch_size num_samples action_dim']
        sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.action_dim)
        best_actions = torch.gather(sample_actions, dim=1, index=best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch: Float[torch.Tensor, 'batch_size*top_actions state_dim']
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)

        best_actions: Float[torch.Tensor, 'batch_size*num_samples action_dim']
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        return state_batch, repeated_states, sample_actions, sorted_q_inds, stacked_s_batch, best_actions, batch_size

    def compute_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        gamma_exp_batch = batch.gamma_exponent
        dp_mask = batch.boot_decision_point

        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

        if self.ensemble_targets:
            _, next_q = self.q_critic.get_qs_target(
                next_state_batch, next_actions,
            )
        else:
            next_q = self.q_critic.get_q_target(next_state_batch, next_actions)

        target = reward_batch + mask_batch * (self.gamma ** gamma_exp_batch) * next_q
        _, q_ens = self.q_critic.get_qs(state_batch, action_batch, with_grad=True)
        return ensemble_mse(target, q_ens)

    def compute_sampler_entropy_loss(self, update_info: tuple) -> torch.Tensor:
        (_, repeated_states, sample_actions, _, stacked_s_batch, best_actions, batch_size) = update_info

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

    def compute_sampler_no_entropy_loss(self, update_info: tuple) -> torch.Tensor:
        (state_batch, _, sample_actions, sorted_q_inds, _, _, batch_size) = update_info

        # gets the best actions
        best_ind_proposal = sorted_q_inds[:, :self.top_actions_proposal]
        best_ind_proposal = best_ind_proposal.repeat_interleave(self.action_dim, -1)
        best_actions_proposal = torch.gather(sample_actions, 1, best_ind_proposal)
        stacked_s_batch_proposal = state_batch.repeat_interleave(self.top_actions_proposal, dim=0)
        best_actions_proposal = torch.reshape(best_actions_proposal, (-1, self.action_dim))

        logp, _ = self.sampler.get_log_prob(stacked_s_batch_proposal, best_actions_proposal, with_grad=True)
        sampler_loss = logp.reshape(batch_size, self.top_actions_proposal, 1)
        sampler_loss = -1 * (sampler_loss.mean(axis=1)).mean()

        return sampler_loss

    def compute_actor_loss(self, update_info) -> (torch.Tensor, tuple):
        _, _, _, _, stacked_s_batch, best_actions, _ = update_info
        logp, _ = self.actor.get_log_prob(stacked_s_batch, best_actions, with_grad=True)
        actor_loss = -logp.mean()  # BUG: This is negative?
        return actor_loss

    def compute_sampler_loss(self, update_info) -> (torch.Tensor, torch.Tensor):
        if self.tau != 0:  # Entropy version of proposal policy update
            sampler_loss = self.compute_sampler_entropy_loss(update_info)
        else:
            # Non-entropy version of proposal policy update.
            # A greater percentage of actions are used to update the proposal policy than the actor policy
            sampler_loss = self.compute_sampler_no_entropy_loss(update_info)

        return sampler_loss

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            batch = self.buffer.sample()
            q_loss = self.compute_critic_loss(batch)
            self.q_critic.update(q_loss)

    def update_actor(self) -> None:
        update_infos = []
        for _ in range(self.n_actor_updates):
            batch = self.buffer.sample()
            update_info = self.get_policy_update_info(batch.state)
            actor_loss = self.compute_actor_loss(update_info)
            self.actor.update(actor_loss)
            update_infos.append(update_info)
        return update_infos

    def update_sampler(self, update_infos: Optional[list[tuple]]) -> None:
        if update_infos is not None:
            for update_info in update_infos:
                sampler_loss = self.compute_sampler_loss(update_info)
                self.sampler.update(sampler_loss)
        else:
            for i in range(self.n_sampler_updates):
                batch = self.buffer.sample()
                update_info = self.get_policy_update_info(batch.state)
                sampler_loss = self.compute_sampler_loss(update_info)
                self.sampler.update(sampler_loss)

    def update(self) -> None:
        # share_batch ensures that update_actor and update_sampler use the same batch
        self.update_critic()
        update_infos = self.update_actor()
        if not self.uniform_proposal:
            if self.share_batch:
                self.update_sampler(update_infos=update_infos)
            else:
                self.update_sampler(update_infos=None)

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


class GreedyACLineSearch(GreedyAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)

        self.actor.set_parameters(id(self.buffer), eval_error_fn=self.actor_eval_error_fn)
        self.sampler.set_parameters(id(self.buffer), eval_error_fn=self.sampler_eval_error_fn)
        self.q_critic.set_parameters(id(self.buffer), eval_error_fn=self.critic_eval_error_fn)

    def critic_eval_error_fn(self, args: list[torch.Tensor]) -> torch.Tensor:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = args
        q = self.q_critic.get_q(state_batch, action_batch, with_grad=False)
        next_action, _ = self.actor.get_action(next_state_batch, with_grad=False)
        next_q = self.q_critic.get_q_target(next_state_batch, next_action)
        target = reward_batch + mask_batch * self.gamma * next_q
        error = torch.nn.functional.mse_loss(q.detach(), target.detach())
        return error

    def actor_eval_error_fn(self, args: list[torch.Tensor]) -> torch.Tensor:
        state_batch, _, _, _, _ = args
        _, _, _, _, stacked_s_batch, best_actions, _ = self.get_policy_update_info(state_batch)
        logp, _ = self.actor.get_log_prob(stacked_s_batch.detach(), best_actions.detach(), with_grad=False)
        return -logp.mean().detach()

    def sampler_eval_error_fn(self, args: list[torch.Tensor]) -> torch.Tensor:
        state_batch, _, _, _, _ = args
        _, _, _, _, stacked_s_batch, best_actions, _ = self.get_policy_update_info(state_batch)
        logp, _ = self.sampler.get_log_prob(stacked_s_batch.detach(), best_actions.detach(), with_grad=False)
        return -logp.mean().detach()


class ExploreLSGAC(GreedyACLineSearch):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        # initialize exploration module
        self.exploration = init_exploration_module(cfg.exploration, state_dim, action_dim)
        self.exploration.set_parameters(id(self.buffer))
        self.exploration_weight = cfg.exploration_weight

    def update(self) -> None:
        super().update()
        # update exploration module
        self.exploration.update()

    def sort_q_value(self, repeated_states: torch.Tensor, sample_actions: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
        q_values = self.q_critic.get_q(repeated_states, sample_actions, with_grad=False)

        # query the exploration bonus
        exp_b = self.exploration.get_exploration_bonus(repeated_states, sample_actions)
        q_values += self.exploration_weight * exp_b

        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q
