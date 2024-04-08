from omegaconf import DictConfig
from pathlib import Path

import torch.nn as nn
import numpy
import os
import pickle as pkl

from root.agent.base import BaseAC
from root.component.actor.factory import init_actor
from root.component.critic.factory import init_v_critic
from root.component.buffer.factory import init_buffer
from root.component.network.utils import to_np, state_to_tensor


class SimpleAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.tau = cfg.tau
        self.device = cfg.device
        self.critic = init_v_critic(cfg.critic, state_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, self.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: tuple) -> None:
        self.buffer.feed(transition)

    def update_actor(self) -> None:
        batch = self.buffer.sample()
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        rewards = batch['rewards']
        dones = batch['dones']

        log_prob, _ = self.actor.get_log_prob(states, actions)
        v = self.critic.get_v(states, with_grad=True)
        v_next = self.critic.get_v(next_states, with_grad=False)
        target = rewards + self.gamma * (1.0 - dones) * v_next
        ent = -log_prob
        loss_actor = -(self.tau * ent + log_prob * (target - v.detach())).mean()

        self.actor.update(loss_actor)

    def update_critic(self) -> None:
        batch = self.buffer.sample()
        states = batch['states']
        next_states = batch['next_states']
        rewards = batch['rewards']
        dones = batch['dones']

        v = self.critic.get_v(states, with_grad=True)
        v_next = self.critic.get_v(next_states, with_grad=False)
        target = rewards + self.gamma * (1.0 - dones) * v_next

        loss_critic = nn.functional.mse_loss(v, target)

        self.critic.update(loss_critic)

    def update(self) -> None:
        self.update_critic()
        self.update_critic()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        critic_path = path / "critic"
        self.critic.save(critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        critic_path = path / "critic"
        self.critic.load(critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)
