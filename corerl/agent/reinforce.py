from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy as np

from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_v_critic
from corerl.component.network.utils import to_np, state_to_tensor, tensor, ensemble_mse
from corerl.utils.device import device
from corerl.data import TransitionBatch, Transition

class Reinforce(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.v_critic = init_v_critic(cfg.critic, state_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)

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
        tensor_state = state_to_tensor(state, device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: Transition) -> None:
        return

    def compute_returns(self) -> None:
        ep_t = len(self.ep_states) - 1
        curr_return = 0.0

        # If the episode is truncated, returns bootstrap the final state
        if self.trunc:
            tensor_state = state_to_tensor(self.ep_states[ep_t], device)
            v_boot = self.v_critic.get_v(tensor_state, with_grad=False)
            curr_return = v_boot

        self.returns = np.zeros(ep_t)

        ep_t -= 1
        for t in range(ep_t, -1, -1):
            curr_return = self.ep_rewards[t] + self.gamma * curr_return
            self.returns[t] = curr_return

        self.returns = tensor(self.returns, device)

        self.ep_states = np.asarray(self.ep_states[:-1])
        self.ep_states = tensor(self.ep_states, device)
        self.ep_actions = np.asarray(self.ep_actions)
        self.ep_actions = tensor(self.ep_actions, device)

    def compute_v_loss(self) -> torch.Tensor:
        _, v_ens = self.v_critic.get_vs(self.ep_states, with_grad=True)
        v_base_loss = ensemble_mse(self.returns, v_ens)

        return v_base_loss

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            v_loss = self.compute_v_loss()
            self.v_critic.update(v_loss)

    def compute_actor_loss(self) -> torch.Tensor:
        v_base = self.v_critic.get_v(self.ep_states, with_grad=False)
        with torch.no_grad():
            delta = self.returns - v_base
        log_prob, _ = self.actor.get_log_prob(self.ep_states, self.ep_actions)
        actor_loss = torch.mean(-log_prob * delta)

        return actor_loss

    def update_actor(self) -> None:
        for _ in range(self.n_actor_updates):
            actor_loss = self.compute_actor_loss()
            self.actor.update(actor_loss)

    def update(self) -> None:
        self.compute_returns()
        self.update_critic()
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