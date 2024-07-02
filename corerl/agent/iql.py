from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy
import pickle as pkl

from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic, init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor, ensemble_expectile_loss, ensemble_mse
from corerl.utils.device import device
from corerl.data.data import TransitionBatch, Transition


class IQL(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.temp = cfg.temp
        self.expectile = cfg.expectile

        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        tensor_state = state_to_tensor(state, device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def compute_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states = batch.state
        actions = batch.action
        v = self.v_critic.get_v(states, with_grad=False)
        q = self.q_critic.get_q(states, actions, with_grad=False)  # NOTE: we are not using target networks
        exp_a = torch.exp((q - v) * self.temp)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device))
        log_probs, _ = self.actor.get_log_prob(states, actions, with_grad=True)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def compute_v_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states, actions = batch.state, batch.action
        q = self.q_critic.get_q(states, actions, with_grad=False)
        _, vs = self.v_critic.get_vs(states, with_grad=True)
        value_loss = ensemble_expectile_loss(q, vs, self.expectile)
        return value_loss

    def compute_q_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states, actions, rewards, next_states, dones, gamma_exps = (batch.state, batch.action,
                                                                    batch.n_step_reward, batch.boot_state,
                                                                    batch.terminated, batch.gamma_exponent)

        next_v = self.v_critic.get_v_target(next_states)
        target = rewards + ((self.gamma ** gamma_exps) * (1 - dones) * next_v)
        _, q_ens = self.q_critic.get_qs(states, actions, with_grad=True)
        q_loss = ensemble_mse(target, q_ens)

        return q_loss

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            batch = self.buffer.sample()

            v_loss = self.compute_v_loss(batch)
            self.v_critic.update(v_loss)

            q_loss = self.compute_q_loss(batch)
            self.q_critic.update(q_loss)

    def update_actor(self) -> None:
        for _ in range(self.n_actor_updates):
            batch = self.buffer.sample()
            actor_loss = self.compute_actor_loss(batch)
            self.actor.update(actor_loss)

    def update(self) -> None:
        self.update_critic()
        self.update_actor()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

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

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)
