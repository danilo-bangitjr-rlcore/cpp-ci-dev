import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from corerl.messages.events import EventType
from corerl.utils.hook import when
import torch
import numpy
import pickle as pkl
import logging
log = logging.getLogger(__name__)
from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor
from corerl.component.exploration.factory import init_exploration_module
from corerl.utils.device import device
from corerl.data.data import TransitionBatch, Transition
import corerl.utils.freezer as fr
from jaxtyping import Float
from typing import Optional
import logging

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


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

        self.uniform_sampling_percentage = cfg.uniform_sampling_percentage
        self.learned_proposal_percent = 1 - self.uniform_sampling_percentage
        self.uniform_proposal = self.uniform_sampling_percentage == 1

        self.n_sampler_updates = cfg.n_sampler_updates
        if self.share_batch and not self.uniform_proposal:
            assert self.n_actor_updates == self.n_sampler_updates, "Actor and proposal must use same number of updates"

        self.top_actions = int(self.rho * self.num_samples)  # Number of actions used to update actor
        self.top_actions_proposal = int(
            self.rho_proposal * self.num_samples)  # Number of actions used to update proposal policy

        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.sampler = init_actor(cfg.actor, state_dim, action_dim, initializer=self.actor)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer)
        self.policy_buffer = init_buffer(cfg.actor.buffer)

        self.sample_all_discrete_actions = False
        if self.discrete_control:
            self.top_actions = 1
            if self.num_samples >= self.action_dim:
                self.num_samples = self.action_dim
                self.top_actions_proposal = self.action_dim

        self._hooks(when.Agent.AfterCreate, self)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        self._msg_bus.emit_event_sync(EventType.agent_get_action)

        tensor_state = state_to_tensor(state, device.device)

        args, _ = self._hooks(when.Agent.BeforeGetAction, self, tensor_state)
        tensor_state = args[1]
        tensor_action, action_info = self.actor.get_action(
            tensor_state, with_grad=False,
        )

        args, _ = self._hooks(
            when.Agent.AfterGetAction, self, tensor_state, tensor_action,
        )
        tensor_state, tensor_action = args[1:]
        action = to_np(tensor_action)[0]

        # log the action_info to the freezer
        fr.freezer.store('action_info', action_info)
        return action

    def update_buffer(self, transition: Transition) -> None:
        self._msg_bus.emit_event_sync(EventType.agent_update_buffer)

        args, _ = self._hooks(
            when.Agent.BeforeUpdateCriticBuffer, self, transition,
        )
        critic_transition = args[1]
        self.critic_buffer.feed(critic_transition)

        # Only train policy on states at decision points
        if transition.state_dp:
            args, _ = self._hooks(
                when.Agent.BeforeUpdateActorBuffer, self, transition,
            )
            actor_transition = args[1]
            self.policy_buffer.feed(actor_transition)

    def load_buffer(self, transitions: list[Transition]) -> None:
        policy_transitions = []
        for transition in transitions:
            if transition.state_dp:
                policy_transitions.append(transition)

        self.policy_buffer.load(policy_transitions)
        self.critic_buffer.load(transitions)

    def get_sorted_q_values(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        sample_actions: Float[torch.Tensor, "batch_size num_samples action_dim"],
    ) -> Float[torch.Tensor, "batch_size num_samples 1"]:
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py

        batch_size = state_batch.shape[0]
        num_samples = sample_actions.shape[1]
        action_dim = sample_actions.shape[2]

        repeated_states = state_batch.repeat_interleave(num_samples, dim=0)
        flattened_actions = sample_actions.view(batch_size * num_samples, action_dim)
        # flattened_actions = sample_actions.reshape(batch_size * num_samples, action_dim)

        q_values: Float[torch.Tensor, "batch_size*num_samples 1"]
        q_values = self.q_critic.get_q(
            [repeated_states], [flattened_actions], with_grad=False, bootstrap_reduct=False
        )
        q_values = q_values.view(batch_size, num_samples, 1)
        sorted_q_inds: Float[torch.Tensor, "batch_size num_samples 1"]
        sorted_q_inds = torch.argsort(q_values, dim=1, descending=True)

        return sorted_q_inds

    def get_uniform_action_samples(
        self, batch_size: int, n_rand: int
    ) -> Float[torch.Tensor, "batch_size n_rand action_dim"]:
        if self.discrete_control:
            action_indices = torch.randint(high=self.action_dim, size=(batch_size, n_rand, 1))
            empty_action_onehot = torch.zeros(size=(batch_size, n_rand, self.action_dim))
            uniform_sample_actions = empty_action_onehot.scatter_(dim=-1, index=action_indices, value=1)
        else:
            uniform_sample_actions = torch.rand(batch_size, n_rand, self.action_dim)

        return uniform_sample_actions.to(device.device)

    def get_sampled_actions(
        self, state_batch: Float[torch.Tensor, "batch_size state_dim"]
    ) -> Float[torch.Tensor, "batch_size num_samples action_dim"]:
        batch_size = state_batch.shape[0]

        if self.uniform_proposal:
            self.uniform_sampling_percentage = 1.0

        if self.learned_proposal_percent > 0:
            n_proposal = int(np.floor(self.num_samples * self.learned_proposal_percent))
            repeated_states: Float[torch.Tensor, "batch_size*n_proposal state_dim"]
            repeated_states = state_batch.repeat_interleave(n_proposal, dim=0)

            proposed_actions: Float[torch.Tensor, "batch_size*n_proposal action_dim"]
            proposed_actions, _ = self.sampler.get_action(
                repeated_states,
                with_grad=False,
            )
            proposed_actions = proposed_actions.reshape(
                batch_size, n_proposal, self.action_dim
            )

        else:
            proposed_actions = torch.empty(batch_size, 0, self.action_dim)

        if self.uniform_sampling_percentage > 0:
            n_rand = int(np.ceil(self.num_samples * self.uniform_sampling_percentage))
            uniform_sample_actions = self.get_uniform_action_samples(batch_size=batch_size, n_rand=n_rand)

        else:
            uniform_sample_actions = torch.empty(batch_size, 0, self.action_dim)

        SAMPLE_DIM = 1
        sample_actions = torch.cat(
            [proposed_actions, uniform_sample_actions], dim=SAMPLE_DIM
        )
        logger.debug(f"{proposed_actions.shape=}")
        logger.debug(f"{uniform_sample_actions.shape=}")

        return sample_actions

    def get_top_actions(
        self,
        proposed_actions: Float[torch.Tensor, "batch_size num_samples action_dim"],
        sorted_q_inds: Float[torch.Tensor, "batch_size num_samples 1"],
        n_top_actions: int,
    ) -> Float[torch.Tensor, "batch_size*n_top_actions action_dim"]:

        best_ind: Float[torch.Tensor, "batch_size n_top_actions action_dim"]
        best_ind = sorted_q_inds[:, :n_top_actions].repeat_interleave(
            self.action_dim, -1
        )

        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        best_actions = torch.gather(proposed_actions, dim=1, index=best_ind)
        batch_size = proposed_actions.shape[0]
        best_actions = torch.reshape(best_actions, (batch_size*n_top_actions, self.action_dim))

        return best_actions

    def get_policy_update_info(self, state_batch: Float[torch.Tensor, 'batch_size state_dim']) -> tuple[
            Float[torch.Tensor, 'batch_size state_dim'],
            Float[torch.Tensor, 'batch_size*num_samples state_dim'],
            Float[torch.Tensor, 'batch_size num_samples action_dim'],
            Float[torch.Tensor, 'batch_size num_samples'],
            Float[torch.Tensor, 'batch_size*top_actions state_dim'],
            Float[torch.Tensor, 'batch_size*num_samples action_dim'],
            int]:

        batch_size = state_batch.shape[0]
        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        repeated_states: Float[torch.Tensor, 'batch_size*num_samples state_dim']
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        sample_actions = self.get_sampled_actions(state_batch)
        sorted_q_inds = self.get_sorted_q_values(state_batch, sample_actions)
        best_actions = self.get_top_actions(sample_actions, sorted_q_inds, n_top_actions=self.top_actions)

        # Reshape samples for calculating the loss
        stacked_s_batch: Float[torch.Tensor, 'batch_size*top_actions state_dim']
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)

        return state_batch, repeated_states, sample_actions, sorted_q_inds, stacked_s_batch, best_actions, batch_size

    def compute_critic_loss(self, ensemble_batch: list[TransitionBatch]) -> list[torch.Tensor]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        next_action_batches = []
        mask_batches = []
        gamma_exp_batches = []
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.state
            action_batch = batch.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.boot_state
            mask_batch = 1 - batch.terminated
            gamma_exp_batch = batch.gamma_exponent
            dp_mask = batch.boot_state_dp

            next_actions, _ = self.actor.get_action(next_state_batch, with_grad=False)
            # For the 'Anytime' paradigm, only states at decision points can sample next_actions
            # If a state isn't at a decision point, its next_action is set to the current action
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

            # Option 1: Using the reduction of the ensemble in the update target
            if not self.ensemble_targets:
                next_q = self.q_critic.get_q_target([next_state_batch], [next_actions], bootstrap_reduct=True)
                next_qs.append(next_q)

            state_batches.append(state_batch)
            action_batches.append(action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            next_action_batches.append(next_actions)
            mask_batches.append(mask_batch)
            gamma_exp_batches.append(gamma_exp_batch)

        # Option 2: Using the corresponding target function in the ensemble in the update target
        if self.ensemble_targets:
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches, bootstrap_reduct=True)
        else:
            for i in range(ensemble):
                next_qs[i] = torch.unsqueeze(next_qs[i], 0)
            next_qs = torch.cat(next_qs, dim=0)

        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True, bootstrap_reduct=True)
        losses = []
        for i in range(ensemble):
            # N-Step SARSA update with variable 'N', thus 'reward_batch' is an n_step reward
            # and the exponent on gamma, 'gamma_exp_batch', depends on 'n'
            target = reward_batches[i] + mask_batches[i] * (self.gamma ** gamma_exp_batches[i]) * next_qs[i]

            args, _ = self._hooks(
              when.Agent.BeforeCriticLossComputed, self, ensemble_batch[i], target, qs[i], i
            )
            # Producing errors when gradient is computed
            #ensemble_batch[i], target, qs[i], _ = args[1:]

            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        return losses

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

    def compute_actor_loss(self, update_info) -> torch.Tensor:
        _, _, _, _, stacked_s_batch, best_actions, _ = update_info
        logp, _ = self.actor.get_log_prob(stacked_s_batch, best_actions, with_grad=True)
        actor_loss = -logp.mean()  # BUG: This is negative?
        return actor_loss

    def compute_sampler_loss(self, update_info) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tau != 0:  # Entropy version of proposal policy update
            sampler_loss = self.compute_sampler_entropy_loss(update_info)
        else:
            # Non-entropy version of proposal policy update.
            # A greater percentage of actions are used to update the proposal policy than the actor policy
            sampler_loss = self.compute_sampler_no_entropy_loss(update_info)

        return sampler_loss

    def update_critic(self) -> None:
        self._msg_bus.emit_event_sync(EventType.agent_update_critic)

        for _ in range(self.n_critic_updates):
            batches = self.critic_buffer.sample()
            args, _ = self._hooks(
                when.Agent.AfterCriticBufferSample, self, batches,
            )
            batches = args[1]

            def closure():
                return sum(self.compute_critic_loss(batches))
            q_loss = closure()

            args, _ = self._hooks(
                when.Agent.AfterCriticLossComputed, self, batches, q_loss,
            )
            batches, q_loss = args[1:]
            self.q_critic.update(q_loss, opt_kwargs={"closure": closure})

            args, _ = self._hooks(
                when.Agent.AfterCriticUpdate, self, batches, q_loss,
            )
            batches, q_loss = args[1:]

        return q_loss

    def update_actor(self) -> None:
        self._msg_bus.emit_event_sync(EventType.agent_update_actor)

        update_infos = []
        for _ in range(self.n_actor_updates):
            batches = self.policy_buffer.sample()
            # Assuming we don't have an ensemble of policies
            assert len(batches) == 1
            batch = batches[0]

            args, _ = self._hooks(
                when.Agent.AfterActorBufferSample, self, batch,
            )
            batch = args[1]

            update_info = self.get_policy_update_info(batch.state)
            args, _ = self._hooks(
                when.Agent.BeforeActorLossComputed, self, update_info,
            )
            update_info = args[1]

            actor_loss = self.compute_actor_loss(update_info)
            args, _ = self._hooks(
                when.Agent.AfterActorLossComputed, self, batch, update_info,
                actor_loss,
            )
            batch, update_info, actor_loss = args[1:]

            stacked_s_batch = update_info[4]
            best_actions = update_info[5]
            self.actor.update(
                actor_loss,
                opt_kwargs={
                    "closure": lambda: self.actor_err(
                        stacked_s_batch, best_actions,
                    ),
                },
            )
            update_infos.append(update_info)

            args, _ = self._hooks(
                when.Agent.AfterActorUpdate, self, batch, actor_loss,
            )
            actor_loss = args[1]

        return update_infos

    def update_sampler(self, update_infos: Optional[list[tuple]]) -> None:
        if update_infos is not None:
            for update_info in update_infos:
                args, _ = self._hooks(
                    when.Agent.BeforeProposalLossComputed, self, update_info,
                )
                update_info = args[1]

                sampler_loss = self.compute_sampler_loss(update_info)
                args, _ = self._hooks(
                    when.Agent.AfterProposalLossComputed, self, None,
                    update_info, sampler_loss,
                )
                batch, update_info, sampler_loss = args[1:]

                stacked_s_batch = update_info[4]
                best_actions = update_info[5]
                self.sampler.update(
                    sampler_loss,
                    opt_kwargs={
                        "closure": lambda: self.sampler_err(
                            stacked_s_batch, best_actions,
                        ),
                    },
                )

                self._hooks(
                    when.Agent.AfterProposalUpdate, self, batch, sampler_loss,
                )
        else:
            for i in range(self.n_sampler_updates):
                batches = self.policy_buffer.sample()
                assert len(batches) == 1
                batch = batches[0]

                args, _ = self._hooks(
                    when.Agent.AfterProposalBufferSample, self, batch,
                )
                batch = args[1]

                update_info = self.get_policy_update_info(batch.state)
                args, _ = self._hooks(
                    when.Agent.BeforeProposalLossComputed, self, update_info,
                )
                update_info = args[1]

                sampler_loss = self.compute_sampler_loss(update_info)
                args, _ = self._hooks(
                    when.Agent.AfterProposalLossComputed, self, batch,
                    update_info, sampler_loss,
                )
                batch, update_info, sampler_loss = args[1:]

                stacked_s_batch = update_info[4]
                best_actions = update_info[5]
                self.sampler.update(
                    sampler_loss,
                    opt_kwargs={
                        "closure": lambda: self.sampler_err(
                            stacked_s_batch, best_actions,
                        ),
                    },
                )

                self._hooks(
                    when.Agent.AfterProposalUpdate, self, batch, sampler_loss,
                )

    def actor_err(self, stacked_s_batch, best_actions) -> torch.Tensor:
        logp, _ = self.actor.get_log_prob(
            stacked_s_batch, best_actions, with_grad=True,
        )
        return -logp.mean()

    def sampler_err(self, stacked_s_batch, best_actions) -> torch.Tensor:
        logp, _ = self.sampler.get_log_prob(
            stacked_s_batch, best_actions, with_grad=True,
        )
        return -logp.mean()

    def update(self) -> None:
        # share_batch ensures that update_actor and update_sampler use the same batch
        critic_loss = None
        if min(self.critic_buffer.size) > 0:
            critic_loss = self.update_critic()

        if min(self.policy_buffer.size) > 0:
            update_infos = self.update_actor()
            if not self.uniform_proposal:
                if self.share_batch:
                    self.update_sampler(update_infos=update_infos)
                else:
                    self.update_sampler(update_infos=None)

        return critic_loss

    def save(self, path: Path) -> None:
        self._msg_bus.emit_event_sync(EventType.agent_save)

        path.mkdir(parents=True, exist_ok=True)
        actor_path = path / "actor"
        self.actor.save(actor_path)

        sampler_path = path / "sampler"
        self.sampler.save(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "wb") as f:
            pkl.dump(self.policy_buffer, f)

    def load(self, path: Path) -> None:
        self._msg_bus.emit_event_sync(EventType.agent_load)

        actor_path = path / "actor"
        self.actor.load(actor_path)

        sampler_path = path / "sampler"
        self.sampler.load(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "rb") as f:
            self.policy_buffer = pkl.load(f)

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {
            'critic': self.critic_buffer.size,
            'policy': self.policy_buffer.size,
        }


class GreedyACLineSearch(GreedyAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)

        self.actor.set_parameters(id(self.policy_buffer), eval_error_fn=self.actor_eval_error_fn)
        self.sampler.set_parameters(id(self.policy_buffer), eval_error_fn=self.sampler_eval_error_fn)
        self.q_critic.set_parameters(id(self.critic_buffer), eval_error_fn=self.critic_eval_error_fn)

    def critic_eval_error_fn(self, args: list[torch.Tensor]) -> torch.Tensor:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = args
        q = self.q_critic.get_q([state_batch], [action_batch], with_grad=False, bootstrap_reduct=True)
        next_action, _ = self.actor.get_action(next_state_batch, with_grad=False)
        next_q = self.q_critic.get_q_target([next_state_batch], [next_action], bootstrap_reduct=True)
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
        self.exploration.set_parameters(id(self.critic_buffer))
        self.exploration_weight = cfg.exploration_weight

    def update(self) -> None:
        super().update()
        # update exploration module
        self.exploration.update()

    def sort_q_value(self, repeated_states: torch.Tensor, sample_actions: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
        q_values = self.q_critic.get_q([repeated_states], [sample_actions], with_grad=False, bootstrap_reduct=True)

        # query the exploration bonus
        exp_b = self.exploration.get_exploration_bonus(repeated_states, sample_actions)
        q_values += self.exploration_weight * exp_b

        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q
