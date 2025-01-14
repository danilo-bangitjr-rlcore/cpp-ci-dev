from collections.abc import Sequence
from dataclasses import field
from typing import Literal
from pathlib import Path

import torch
import numpy
import random
import pickle as pkl

from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.configs.config import config
from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor
from corerl.state import AppState
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import TransitionBatch, Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions


@config(frozen=True)
class EpsilonGreedySarsaConfig(BaseAgentConfig):
    name: Literal['epsilon_greedy_sarsa'] = 'epsilon_greedy_sarsa'

    ensemble_targets: bool = False
    epsilon: float = 0.1
    samples: int = 10_000

    critic: EnsembleCriticConfig = field(default_factory=EnsembleCriticConfig)


class EpsilonGreedySarsa(BaseAgent):
    def __init__(self, cfg: EpsilonGreedySarsaConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.ensemble_targets = cfg.ensemble_targets
        self.samples = cfg.samples
        self.epsilon = cfg.epsilon
        self.action_dim = self.action_dim
        self.q_critic = init_q_critic(cfg.critic, self.state_dim, self.action_dim)
        self.critic_buffer = init_buffer(cfg.critic.buffer)

    def update_buffer(self, transitions: Sequence[Transition]) -> None:
        self.critic_buffer.feed(transitions)

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
        gamma_batches = []
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.prior.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

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
            target = reward_batches[i] + gamma_batches[i] * next_qs[i]
            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        return losses

    def atomic_critic_update(self) -> float:
        batches = self.critic_buffer.sample()
        q_loss = self.compute_q_loss(batches)
        self.q_critic.update(q_loss)

        float_losses = [float(loss) for loss in q_loss]

        return sum(float_losses) / len(float_losses)

    def update(self) -> list[float]:
        critic_losses = []
        if min(self.critic_buffer.size) > 0:
            for _ in range(self.n_updates):
                critic_loss = self.atomic_critic_update()
                critic_losses.append(critic_loss)

        return critic_losses

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

    def load_buffer(self, transitions: Sequence[Transition]) -> None:
        ...
