from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING
from pathlib import Path

import torch
import numpy
import random
import pickle as pkl

from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import TransitionBatch, Transition


@dataclass
class EpsilonGreedySarsaConfig(BaseAgentConfig):
    name: str = 'epsilon_greedy_sarsa'

    ensemble_targets: bool = False
    epsilon: float = 0.1
    samples: int = 10_000

    critic: Any = MISSING

    defaults: list[Any] = field(default_factory=lambda: [
        'base_agent',
        { 'critic': 'critic' },
        '_self_',
    ])


class EpsilonGreedySarsa(BaseAgent):
    def __init__(self, cfg: EpsilonGreedySarsaConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.ensemble_targets = cfg.ensemble_targets
        self.samples = cfg.samples
        self.epsilon = cfg.epsilon
        self.action_dim = action_dim
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.critic_buffer = init_buffer(cfg.critic.buffer)

    def update_buffer(self, transition: Transition) -> None:
        self.critic_buffer.feed(transition)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device.device)
        action_np = to_np(self._get_action(tensor_state))[0]
        return action_np

    def _get_action(self, state: torch.Tensor) -> torch.Tensor:
        num_states = state.shape[0]
        actions = torch.zeros(num_states, self.action_dim)
        for i, o in enumerate(state):
            if random.random() <= self.epsilon:
                action = torch.rand((1, self.action_dim), device=device.device)
            else:
                action = self.get_greedy_action(o)
            actions[i, :] = action

        return actions

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.unsqueeze(state, dim=0)
        state_repeated = torch.repeat_interleave(state, self.samples, dim=0)
        action_samples = torch.rand((self.samples, self.action_dim))
        q = self.q_critic.get_q([state_repeated], [action_samples], with_grad=False)
        max_q_idx = torch.argmax(q)
        greedy_action = action_samples[max_q_idx, :]
        return greedy_action

    def compute_q_loss(self, ensemble_batch: list[TransitionBatch]) -> list[torch.Tensor]:
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

            next_actions = self._get_action(next_state_batch)
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

            # Option 1: Using the reduction of the ensemble in the update target
            if not self.ensemble_targets:
                next_q = self.q_critic.get_q_target([next_state_batch], [next_actions])
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
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches)
        else:
            for i in range(ensemble):
                next_qs[i] = torch.unsqueeze(next_qs[i], 0)
            next_qs = torch.cat(next_qs, dim=0)

        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            target = reward_batches[i] + mask_batches[i] * (self.gamma ** gamma_exp_batches[i]) * next_qs[i]
            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        return losses

    def atomic_critic_update(self) -> None:
        batches = self.critic_buffer.sample()
        q_loss = self.compute_q_loss(batches)
        self.q_critic.update(q_loss)

    def update(self) -> None:
        if min(self.critic_buffer.size) > 0:
            for _ in range(self.n_updates):
                self.atomic_critic_update()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        critic_path = path / "critic"
        self.q_critic.save(critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

    def load(self, path: Path) -> None:
        critic_path = path / "critic"
        self.q_critic.load(critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

    def load_buffer(self, transitions: list[Transition]) -> None:
        ...
