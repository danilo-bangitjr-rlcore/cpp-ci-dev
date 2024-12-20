from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import numpy
import pickle as pkl

from corerl.configs.config import config
from corerl.agent.base import BaseAC, BaseACConfig
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic, init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor, expectile_loss
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import TransitionBatch, Transition


@config(frozen=True)
class IQLConfig(BaseACConfig):
    name: Literal['iql'] = 'iql'

    temp: float = 1.0
    expectile: float = 0.8


class IQL(BaseAC):
    def __init__(self, cfg: IQLConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.temp = cfg.temp
        self.expectile = cfg.expectile

        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer)
        self.policy_buffer = init_buffer(cfg.actor.buffer)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transitions: Sequence[Transition]) -> None:
        self.critic_buffer.feed(transitions)
        self.policy_buffer.feed([
            t for t in transitions if t.prior.dp
        ])

    def compute_actor_loss(
        self,
        update_info: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    ) -> torch.Tensor:
        _, _, _, _, states, actions, _ = update_info
        v = self.v_critic.get_v([states], with_grad=False)
        q = self.q_critic.get_q([states], [actions], with_grad=False)  # NOTE: we are not using target networks
        exp_a = torch.exp((q - v) * self.temp)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(device.device))
        log_probs, _ = self.actor.get_log_prob(states, actions, with_grad=True)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    """
    def compute_v_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states, actions = batch.state, batch.action
        q = self.q_critic.get_q(states, actions, with_grad=False)
        _, vs = self.v_critic.get_vs(states, with_grad=True)
        value_loss = ensemble_expectile_loss(q, vs, self.expectile)
        return value_loss
    """

    def compute_v_loss(self, ensemble_batch: list[TransitionBatch]) -> list[torch.Tensor]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.prior.action

            state_batches.append(state_batch)
            action_batches.append(action_batch)

        _, qs = self.q_critic.get_qs_target(state_batches, action_batches)
        _, vs = self.v_critic.get_vs(state_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            diff = qs[i] - vs[i]
            loss = expectile_loss(diff, self.expectile)
            losses.append(loss)

        return losses

    def compute_q_loss(self, ensemble_batch: list[TransitionBatch]) -> list[torch.Tensor]:
        ensemble = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        gamma_batches = []
        next_vs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.prior.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma

            state_batches.append(state_batch)
            action_batches.append(action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            gamma_batches.append(gamma_batch)

        _, next_vs = self.v_critic.get_vs(next_state_batches)
        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True)
        losses = []
        for i in range(ensemble):
            target = reward_batches[i] + gamma_batches[i] * next_vs[i]
            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        return losses

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
            empty = torch.empty(0)
            actor_loss = self.compute_actor_loss((
                empty, empty, empty, empty, batch.prior.state, batch.prior.action, 0,
            ))
            self.actor.update(actor_loss)

        return tuple()

    def update(self) -> None:
        if min(self.critic_buffer.size) > 0:
            self.update_critic()
        if min(self.policy_buffer.size) > 0:
            self.update_actor()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

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

    def load_buffer(self, transitions: Sequence[Transition]) -> None:
        ...
