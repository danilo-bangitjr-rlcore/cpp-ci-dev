from omegaconf import DictConfig
from pathlib import Path

import torch.nn as nn
import torch
import numpy
import pickle as pkl

from root.agent.base import BaseAC
from root.component.actor.factory import init_actor
from root.component.critic.factory import init_v_critic, init_q_critic
from root.component.buffer.factory import init_buffer
from root.component.network.utils import to_np, state_to_tensor
from root.component.actor.network_actor import NetworkActor


class InAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.temp = cfg.temp
        self.eps = cfg.eps
        self.exp_threshold = cfg.exp_threshold

        self.device = cfg.device
        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.behaviour = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

    def update_buffer(self, transition: tuple) -> None:
        self.buffer.feed(transition)
    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, self.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def compute_beh_loss(self, data: dict) -> torch.Tensor:
        states, actions = data['states'], data['actions']
        beh_log_probs, _ = self.behaviour.get_log_prob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss

    def compute_v_loss(self, data: dict) -> torch.Tensor:
        states = data['states']
        v_phi = self.v_critic.get_v(states, with_grad=True)
        actions, _ = self.actor.get_action(states, with_grad=False)
        log_probs, _ = self.actor.get_log_prob(states, actions)
        q = self.q_critic.get_q_target(states, actions)
        target = q - self.temp * log_probs
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss

    def compute_q_loss(self, data):
        states, actions, rewards, next_states, dones = (data['states'], data['actions'], data['rewards'],
                                                        data['next_states'], data['dones'])

        q = self.q_critic.get_q(states, actions, with_grad=True)
        next_actions, _ = self.actor.get_action(next_states, with_grad=False)
        next_log_probs, _ = self.actor.get_log_prob(next_states, next_actions,
                                                    with_grad=False)

        q_pi_target = self.q_critic.get_q_target(next_states, next_actions) - self.temp * next_log_probs
        target = rewards + self.gamma * (1 - dones) * q_pi_target
        q_loss = nn.functional.mse_loss(q, target)
        return q_loss

    def compute_actor_loss(self, data):
        states, actions = data['states'], data['actions']
        log_probs, _ = self.actor.get_log_prob(states, actions, with_grad=True)
        q = self.q_critic.get_q(states, actions, with_grad=False)
        v = self.v_critic.get_v(states, with_grad=False)
        beh_log_prob, _ = self.behaviour.get_log_prob(states, actions, with_grad=False)
        clipped = torch.clip(torch.exp((q - v) / self.temp - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss

    def update_critic(self) -> None:
        batch = self.buffer.sample()

        v_loss = self.compute_v_loss(batch)
        self.v_critic.update(v_loss)

        q_loss = self.compute_q_loss(batch)
        self.q_critic.update(q_loss)

    def update_actor(self) -> None:
        batch = self.buffer.sample()
        actor_loss = self.compute_actor_loss(batch)
        self.actor.update(actor_loss)

    def update_beh(self) -> None:
        batch = self.buffer.sample()
        beh_loss = self.compute_beh_loss(batch)
        self.behaviour.update(beh_loss)

    def update(self) -> None:
        self.update_critic()
        self.update_actor()
        # unsure if beh updates should go here. Han pleas advise.

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

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        beh_path = path / "behaviour"
        self.behaviour.load(beh_path)

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)
