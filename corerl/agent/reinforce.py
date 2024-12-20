from pathlib import Path
from typing import Literal

import torch
import numpy as np

from pydantic.dataclasses import dataclass as config
from corerl.agent.base import BaseAC, BaseACConfig
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic
from corerl.component.network.utils import to_np, state_to_tensor, tensor, ensemble_mse
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import NewTransition


@config(frozen=True)
class ReinforceConfig(BaseACConfig):
    name: Literal['reinforce'] = 'reinforce'
    ensemble_targets: bool = False

class Reinforce(BaseAC):
    def __init__(self, cfg: ReinforceConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.ensemble_targets = cfg.ensemble_targets
        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)

        self.ep_states: list[np.ndarray] = []
        self.ep_actions: list[float] = []
        self.ep_rewards: list[float] = []

    # Call at the beginning of episode once environment has been reset
    def reset_episode_stats(self, reset_state: np.ndarray) -> None:
        self.ep_states = [reset_state]
        self.ep_actions = []
        self.ep_rewards = []

    def update_episode_stats(self, action: float, reward: float, next_state: np.ndarray, trunc: bool) -> None:
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_states.append(next_state)
        self.trunc = trunc

    def get_action(self, state: np.ndarray) -> np.ndarray:
        tensor_state = state_to_tensor(state, device.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: NewTransition) -> None:
        return

    def compute_returns(self) -> None:
        ep_t = len(self.ep_states) - 1
        curr_return = 0.0

        # If the episode is truncated, returns bootstrap the final state
        if self.trunc:
            tensor_state = state_to_tensor(self.ep_states[ep_t], device.device)
            if self.ensemble_targets:
                v_boot = self.v_critic.get_vs([tensor_state], with_grad=False)
            else:
                v_boot = self.v_critic.get_v([tensor_state], with_grad=False)
            curr_return = v_boot

        self.returns = np.zeros(ep_t)

        ep_t -= 1
        for t in range(ep_t, -1, -1):
            curr_return = self.ep_rewards[t] + self.gamma * torch.tensor(curr_return)
            self.returns[t] = curr_return

        self.returns = tensor(self.returns, device.device)

        self.ep_states = self.ep_states[:-1]

    def compute_v_loss(self) -> list[torch.Tensor]:
        states = [torch.Tensor(self.ep_states, device.device)]
        _, v_ens = self.v_critic.get_vs(states, with_grad=True)
        v_base_loss = ensemble_mse(self.returns, v_ens)

        return v_base_loss

    def update_critic(self) -> list[float]:
        critic_losses = []
        for _ in range(self.n_critic_updates):
            v_loss = self.compute_v_loss()
            self.v_critic.update(v_loss)

            float_losses = [float(loss) for loss in v_loss]
            critic_losses.append(sum(float_losses) / len(float_losses))

    def compute_actor_loss(self) -> torch.Tensor:
        states = [torch.Tensor(self.ep_states, device.device)]
        v_base = self.v_critic.get_v(states, with_grad=False)
        with torch.no_grad():
            delta = self.returns - v_base

        actions = torch.Tensor(self.ep_actions, device.device)
        log_prob, _ = self.actor.get_log_prob(states[0], actions)
        actor_loss = torch.mean(-log_prob * delta)

        return actor_loss

    def update_actor(self) -> tuple:
        for _ in range(self.n_actor_updates):
            actor_loss = self.compute_actor_loss()
            self.actor.update(actor_loss)

        return tuple()

    def update(self) -> list[float]:
        self.compute_returns()
        critic_loss = self.update_critic()
        self.update_actor()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        v_critic_path = path / "v_critic"
        self.v_critic.save(v_critic_path)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)
