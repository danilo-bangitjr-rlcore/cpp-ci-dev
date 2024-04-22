from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy as np
import pickle as pkl

from root.agent.base import BaseAC
from root.component.actor.factory import init_actor
from root.component.critic.factory import init_v_critic, init_q_critic
from root.component.buffer.factory import init_buffer
from root.component.network.utils import to_np, state_to_tensor, Float, ensemble_mse

class SAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.device = cfg.device

        # self.v_critic = init_v_critic(cfg.critic, state_dim) # Paper has V and Q...
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)
        
        # Entropy
        self.automatic_entropy_tuning = cfg.tau == -1
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_dim).item() # If action_dim is an int, is this necessary?
            self.log_alpha = Float(self.device, 0.0)
        else:
            self.log_alpha = Float(self.device, np.log(cfg.tau))
        self.alpha = self.log_alpha().exp().detach()
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=cfg.lr_alpha)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        tensor_state = state_to_tensor(state, self.device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: tuple) -> None:
        self.buffer.feed(transition)

    def compute_q_loss(self, batch: dict) -> (list, torch.Tensor):
        state_batch = batch['states']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        next_state_batch = batch['next_states']
        mask_batch = 1 - batch['dones']

        next_state_action, info = self.actor.get_action(next_state_batch, with_grad=False)
        next_state_log_pi = info['logp'] # Do this or should we call self.actor.get_log_prob?
        
        q_pi_targ = self.q_critic.get_q_target(next_state_batch, next_state_action)
        q_pi_targ -= self.alpha * next_state_log_pi
        
        # Version of SAC that uses state-value function
        # q_pi_targ = self.v_critic.get_v(next_state_batch, with_grad=False)

        next_q_value = reward_batch + mask_batch * self.gamma * q_pi_targ
        _, q_ens = self.q_critic.get_qs(state_batch, action_batch, with_grad=True)

        critic_loss = ensemble_mse(next_q_value, q_ens)

        return critic_loss

    # Version of SAC that uses state-value function
    # def compute_v_loss(self, batch: dict) -> (torch.Tensor, np.ndarray, np.ndarray):
    #     """L_{\phi}, learn z for state value, v = tau log z"""
    #     state_batch = batch['states']
    #     
    #     v_phi = self.v_critic.get_v(state_batch, with_grad=True)
    #     actions, info = self.actor.get_action(state_batch, with_grad=False)
    #     log_probs = info['logp'] # Do this or should we call self.actor.get_log_prob?
    #     min_q = self.q_critic.get_q(state_batch, actions)
    #     target = min_q - self.alpha * log_probs
    #     value_loss = (0.5 * (v_phi - target) ** 2).mean()
    #
    #     return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()

    def compute_actor_loss(self, batch: dict) -> (torch.Tensor, torch.Tensor):
        state_batch = batch['states']

        actions, info = self.actor.get_action(state_batch, with_grad=True)
        log_pi = info['logp'] # Do this or should we call self.actor.get_log_prob?
        min_q = self.q_critic.get_q(state_batch, actions, with_grad=True)
        policy_loss = ((self.alpha * log_pi) - min_q).mean()

        return policy_loss, log_pi

    def update_entropy(self, batch: dict):
        batch = self.buffer.sample()
        state_batch = batch['states']

        _, info = self.actor.get_action(state_batch, with_grad=False)
        log_pi = info['logp'] # Do this or should we call self.actor.get_log_prob?
        alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha().exp().detach()

    def update_critic(self) -> None:
        batch = self.buffer.sample()

        """
        v_loss = self.compute_v_loss(batch)
        self.v_critic.update(v_loss)
        """

        q_loss = self.compute_q_loss(batch)
        self.q_critic.update(q_loss)

    def update_actor(self) -> None:
        batch = self.buffer.sample()
        actor_loss = self.compute_actor_loss(batch)
        self.actor.update(actor_loss)

    def update(self) -> None:
        self.update_critic()
        self.update_actor()

        if self.automatic_entropy_tuning:
            self.update_entropy()
    
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        """
        v_critic_path = path / "v_critic"
        self.v_critic.save(v_critic_path)
        """

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        log_alpha_path = path / "log_alpha"
        torch.save(self.log_alpha.state_dict(), log_alpha_path)

        alpha_opt_path = path / "alpha_opt"
        torch.save(self.alpha_optimizer.state_dict(), alpha_opt_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        """
        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)
        """

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        log_alpha_path = path / "log_alpha"
        self.log_alpha.load_state_dict(torch.load(log_alpha_path, map_location=self.device))

        alpha_opt_path = path / "alpha_opt"
        self.alpha_optimizer.load_state_dict(torch.load(alpha_opt_path, map_location=self.device))

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)