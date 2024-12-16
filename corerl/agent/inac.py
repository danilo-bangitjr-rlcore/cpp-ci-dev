from pathlib import Path
from typing import Literal

import torch.nn as nn
import torch
import numpy
import pickle as pkl

from corerl.configs.config import config
from corerl.agent.base import BaseAC, BaseACConfig
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic, init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import NewTransitionBatch, NewTransition


@config(frozen=True)
class InACConfig(BaseACConfig):
    name: Literal['inac'] = 'inac'

    ensemble_targets: bool = False
    eps: float = 1e-8
    exp_threshold: int = 10_000
    temp: float = 1.0


class InAC(BaseAC):
    def __init__(self, cfg: InACConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.ensemble_targets = cfg.ensemble_targets
        self.temp = cfg.temp
        self.eps = cfg.eps
        self.exp_threshold = cfg.exp_threshold
        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.behaviour = init_actor(cfg.actor, state_dim, action_dim)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer)
        self.policy_buffer = init_buffer(cfg.actor.buffer)

    def update_buffer(self, transition: NewTransition) -> None:
        self.critic_buffer.feed(transition)
        # Only train policy on states at decision points
        if transition.prior.dp:
            self.policy_buffer.feed(transition)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def compute_beh_loss(self, batch: NewTransitionBatch) -> torch.Tensor:
        states, actions = batch.prior.state, batch.prior.action
        beh_log_probs, _ = self.behaviour.get_log_prob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss

    def compute_v_loss(self, ensemble_batch: list[NewTransitionBatch]) -> list[torch.Tensor]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        log_probs_batches = []
        qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.prior.action
            dp_mask = batch.post.dp

            sampled_actions, _ = self.actor.get_action(state_batch, with_grad=False)
            # Is this what we should be doing for InAC?
            with torch.no_grad():
                actions = (dp_mask * sampled_actions) + ((1.0 - dp_mask) * action_batch)

            log_probs, _ = self.actor.get_log_prob(state_batch, actions)

            # Option 1: Using the reduction of the ensemble in the update target
            if not self.ensemble_targets:
                q = self.q_critic.get_q_target([state_batch], [actions])
                qs.append(q)

            state_batches.append(state_batch)
            action_batches.append(actions)
            log_probs_batches.append(log_probs)

        # Option 2: Using the corresponding target function in the ensemble in the update target
        if self.ensemble_targets:
            _, qs = self.q_critic.get_qs_target(state_batches, action_batches)
        else:
            for i in range(ensemble):
                qs[i] = torch.unsqueeze(qs[i], 0)
            qs = torch.cat(qs, dim=0)

        _, v_phis = self.v_critic.get_vs(state_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            target = qs[i] - self.temp * log_probs_batches[i]
            value_loss = (0.5 * (v_phis[i] - target) ** 2).mean()
            losses.append(value_loss)

        return losses


    def compute_q_loss(self, ensemble_batch: list[NewTransitionBatch]) -> list[torch.Tensor]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        next_action_batches = []
        next_log_probs_batches = []
        gamma_batches = []
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.prior.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

            next_actions, _ = self.actor.get_action(next_state_batch, with_grad=False)
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

            next_log_probs, _ = self.actor.get_log_prob(next_state_batch, next_actions,
                                                        with_grad=False)

            # Option 1: Using the reduction of the ensemble in the update target
            if not self.ensemble_targets:
                next_q = self.q_critic.get_q_target([next_state_batch], [next_actions])
                next_qs.append(next_q)

            state_batches.append(state_batch)
            action_batches.append(action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            next_action_batches.append(next_actions)
            next_log_probs_batches.append(next_log_probs)
            gamma_batches.append(gamma_batch)

        # Option 2: Using the corresponding target function in the ensemble in the update target
        if self.ensemble_targets:
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches)
        else:
            for i in range(ensemble):
                next_qs[i] = torch.unsqueeze(next_qs[i], 0)
            next_qs = torch.cat(next_qs, dim=0)

        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            q_pi_target = next_qs[i] - self.temp * next_log_probs_batches[i]
            target = reward_batches[i] + gamma_batches[i] * q_pi_target
            losses.append(nn.functional.mse_loss(target, qs[i]))

        return losses

    def compute_actor_loss(self, batch: NewTransitionBatch):
        states, actions = batch.prior.state, batch.prior.action
        log_probs, _ = self.actor.get_log_prob(states, actions, with_grad=True)
        q = self.q_critic.get_q([states], [actions], with_grad=False)
        v = self.v_critic.get_v([states], with_grad=False)
        beh_log_prob, _ = self.behaviour.get_log_prob(states, actions, with_grad=False)
        clipped = torch.clip(torch.exp((q - v) / self.temp - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            batches = self.critic_buffer.sample()

            v_loss = self.compute_v_loss(batches)
            self.v_critic.update(v_loss)

            q_loss = self.compute_q_loss(batches)
            self.q_critic.update(q_loss)

    def update_actor(self) -> tuple:
        for _ in range(self.n_actor_updates):
            batches = self.policy_buffer.sample()
            # Assuming we don't have an ensemble of policies
            assert len(batches) == 1
            batch = batches[0]
            actor_loss = self.compute_actor_loss(batch)
            self.actor.update(actor_loss)

        return tuple()

    def update_beh(self) -> None:
        batches = self.policy_buffer.sample()
        # Assuming we don't have an ensemble of policies
        assert len(batches) == 1
        batch = batches[0]
        beh_loss = self.compute_beh_loss(batch)
        self.behaviour.update(beh_loss)

    def update(self) -> None:
        if min(self.critic_buffer.size) > 0:
            self.update_critic()

        if min(self.policy_buffer.size) > 0:
            self.update_actor()
        # unsure if beh updates should go here. Han please advise.

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        beh_path = path / "behaviour"
        self.behaviour.save(beh_path)

        v_critic_path = path / "v_critic"
        self.v_critic.save(v_critic_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "wb") as f:
            pkl.dump(self.policy_buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        beh_path = path / "behaviour"
        self.behaviour.load(beh_path)

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "rb") as f:
            self.policy_buffer = pkl.load(f)

    def load_buffer(self, transitions: list[NewTransition]) -> None:
        ...
