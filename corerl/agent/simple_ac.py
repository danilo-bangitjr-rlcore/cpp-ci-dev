from omegaconf import DictConfig
from pathlib import Path

import numpy
import torch
import pickle as pkl

from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor, ensemble_mse
from corerl.utils.device import device
from corerl.data.data import TransitionBatch, Transition

class SimpleAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.tau = cfg.tau
        self.critic = init_v_critic(cfg.critic, state_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.buffer)
        self.policy_buffer = init_buffer(cfg.buffer)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: Transition) -> None:
        self.critic_buffer.feed(transition)
        # Only train policy on states at decision points
        if transition.state_dp:
            self.policy_buffer.feed(transition)

    def compute_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states = batch.state
        actions = batch.action
        next_states = batch.boot_state
        rewards = batch.n_step_reward
        dones = batch.terminated
        gamma_exps = batch.gamma_exponent

        log_prob, _ = self.actor.get_log_prob(states, actions, with_grad=True)
        v = self.critic.get_v(states, with_grad=False)
        v_next = self.critic.get_v(next_states, with_grad=False)
        target = rewards + (self.gamma ** gamma_exps) * (1.0 - dones) * v_next
        ent = -log_prob
        loss_actor = -(self.tau * ent + log_prob * (target - v.detach())).mean()
        return loss_actor

    def update_actor(self) -> None:
        for _ in range(self.n_actor_updates):
            batch = self.policy_buffer.sample()
            loss_actor = self.compute_actor_loss(batch)
            self.actor.update(loss_actor)

    def compute_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states = batch.state
        next_states = batch.next_state
        rewards = batch.n_step_reward
        dones = batch.terminated
        gamma_exps = batch.gamma_exponent

        _, v_ens = self.critic.get_vs(states, with_grad=True)
        v_next = self.critic.get_v_target(next_states)
        target = rewards + (self.gamma ** gamma_exps) * (1.0 - dones) * v_next

        loss_critic = ensemble_mse(target, v_ens)
        return loss_critic

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            batch = self.critic_buffer.sample()
            loss_critic = self.compute_critic_loss(batch)
            self.critic.update(loss_critic)

    def update(self) -> None:
        if self.critic_buffer.size > 0:
            self.update_critic()
        if self.policy_buffer.size > 0:
            self.update_actor()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        critic_path = path / "critic"
        self.critic.save(critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "wb") as f:
            pkl.dump(self.policy_buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        critic_path = path / "critic"
        self.critic.load(critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "rb") as f:
            self.policy_buffer = pkl.load(f)
