import numpy as np
from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy
import random
import pickle as pkl

from corerl.agent.base import BaseAgent
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, ensemble_mse
from corerl.utils.device import device
from corerl.data import TransitionBatch, Transition

class EpsilonGreedySarsa(BaseAgent):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.samples = cfg.samples
        self.epsilon = cfg.samples
        self.action_dim = action_dim
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def get_action(self, state: torch.Tensor, with_grad=False) -> numpy.ndarray:
        action_np = np.squeeze(to_np(self._get_action(state)))
        return action_np

    def _get_action(self, state: torch.Tensor) -> torch.Tensor:
        num_states = state.shape[0]
        actions = torch.zeros(num_states, self.action_dim)
        for i, o in enumerate(state):
            if random.random() <= self.epsilon:
                action = torch.rand((1, self.action_dim), device=device)
            else:
                action = self.get_greedy_action(o)
            actions[i, :] = action

        return actions

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.unsqueeze(state, dim=0)
        state_repeated = torch.repeat_interleave(state, self.samples)
        action_samples = torch.rand((self.samples, self.action_dim))
        q = self.q_critic.get_q(state_repeated, action_samples, with_grad=False)
        max_q_idx = torch.argmax(q)
        greedy_action = action_samples[max_q_idx, :]
        return greedy_action

    def compute_q_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states, actions, rewards, next_states, dones, gamma_exps, dp_mask = (batch.state, batch.action,
                                                                             batch.reward, batch.next_state, batch.terminated,
                                                                             batch.gamma_exponent, batch.next_decision_point)
        next_actions = self._get_action(next_states)
        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * actions)
        next_q = self.q_critic.get_q_target(next_states, next_actions)
        target = rewards + (1 - dones) * (self.gamma ** gamma_exps) * next_q
        _, q_ens = self.q_critic.get_qs(states, actions, with_grad=True)
        q_loss = ensemble_mse(target, q_ens)
        return q_loss

    def update(self) -> None:
        for _ in range(self.n_updates):
            batch = self.buffer.sample()
            q_loss = self.compute_q_loss(batch)
            self.q_critic.update(q_loss)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        critic_path = path / "critic"
        self.q_critic.save(critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        critic_path = path / "critic"
        self.q_critic.load(critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)
