from abc import ABC, abstractmethod

from base import BaseAC
from root.component.actor.factory import init_actor
from root.component.critic.factory import init_v_critic
from root.component.buffer.factory import init_buffer
import torch.nn as nn


class SimpleAC(BaseAC):
    def __init__(self, cfg, state_dim, action_dim, discrete_control, seed=0):
        super().__init__(cfg, state_dim, action_dim, discrete_control)
        self.critic = init_v_critic(cfg.critic, state_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer, seed)
        self.tau = cfg.tau

    def get_action(self, state):
        return self.actor.get_action(state, with_grad=False)

    def update_buffer(self, transition):
        self.buffer.feed(transition)

    def update_actor(self):
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

    def update_critic(self):
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

    # def save(self):
    #     pass
    #
    # def load(self):
    #     pass
