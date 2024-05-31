from omegaconf import DictConfig
from pathlib import Path

import torch
import numpy as np
import pickle as pkl

from corerl.agent.base import BaseAC
from corerl.component.actor.factory import init_actor
from corerl.component.critic.factory import init_q_critic
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.utils import to_np, state_to_tensor, Float, ensemble_mse
from corerl.utils.device import device
from corerl.data import TransitionBatch, Transition

class SAC(BaseAC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        # self.v_critic = init_v_critic(cfg.critic, state_dim) # Paper has V and Q...
        self.q_critic = init_q_critic(cfg.critic, state_dim, action_dim)
        self.actor = init_actor(cfg.actor, state_dim, action_dim)
        self.buffer = init_buffer(cfg.buffer)
        self.n_entropy_updates = cfg.n_entropy_updates
        
        # Entropy
        self.automatic_entropy_tuning = cfg.tau == -1
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_dim).item() # If action_dim is an int, is this necessary?
            self.log_alpha = Float(device, 0.0)
        else:
            self.log_alpha = Float(device, np.log(cfg.tau))
        self.alpha = self.log_alpha().exp().detach()
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=cfg.lr_alpha)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        tensor_state = state_to_tensor(state, device)
        tensor_action, info = self.actor.get_action(tensor_state, with_grad=False)
        action = to_np(tensor_action)[0]
        return action

    def update_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def compute_q_loss(self, batch: TransitionBatch) -> (list, torch.Tensor):
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        gamma_exp_batch = batch.gamma_exponent
        dp_mask = batch.next_decision_point

        next_actions, info = self.actor.get_action(next_state_batch, with_grad=False)
        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

        next_state_log_pi, _ = self.actor.get_log_prob(next_state_batch, next_actions, with_grad=False)
        
        q_pi_targ = self.q_critic.get_q_target(next_state_batch, next_actions)
        q_pi_targ -= self.alpha * next_state_log_pi
        
        # Version of SAC that uses state-value function
        # q_pi_targ = self.v_critic.get_v(next_state_batch, with_grad=False)
        next_q_value = reward_batch + mask_batch * (self.gamma ** gamma_exp_batch) * q_pi_targ
        _, q_ens = self.q_critic.get_qs(state_batch, action_batch, with_grad=True)

        critic_loss = ensemble_mse(next_q_value, q_ens)

        return critic_loss

    # Version of SAC that uses state-value function
    # def compute_v_loss(self, batch: TransitionBatch) -> (torch.Tensor, np.ndarray, np.ndarray):
    #     """L_{\phi}, learn z for state value, v = tau log z"""
    #     state_batch = batch.state
    #     action_batch = batch.action
    #     dp_mask = batch['state_decision_points']
    #     
    #     v_phi = self.v_critic.get_v(state_batch, with_grad=True)
    #     actions, info = self.actor.get_action(state_batch, with_grad=False)
    #     with torch.no_grad():
    #         actions = (dp_mask * actions) + ((1.0 - dp_mask) * action_batch)
    #     log_probs, _ = self.actor.get_log_prob(state_batch, actions, with_grad=False)
    #     min_q = self.q_critic.get_q(state_batch, actions)
    #     target = min_q - self.alpha * log_probs
    #     value_loss = (0.5 * (v_phi - target) ** 2).mean()
    #
    #     return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()

    def compute_actor_loss(self, batch: TransitionBatch) -> (torch.Tensor, torch.Tensor):
        state_batch = batch.state
        action_batch = batch.action
        dp_mask = batch['state_decision_points']

        actions, info = self.actor.get_action(state_batch, with_grad=True)
        actions = (dp_mask * actions) + ((1.0 - dp_mask) * action_batch)
        log_pi, _ = self.actor.get_log_prob(state_batch, actions, with_grad=True)
        min_q = self.q_critic.get_q(state_batch, actions, with_grad=True)
        policy_loss = ((self.alpha * log_pi) - min_q).mean()

        return policy_loss, log_pi

    def update_entropy(self):
        for _ in range(self.n_entropy_updates):
            batch = self.buffer.sample()
            state_batch = batch.state
            action_batch = batch.action
            dp_mask = batch['state_decision_points']

            actions, info = self.actor.get_action(state_batch, with_grad=False)
            with torch.no_grad():
                actions = (dp_mask * actions) + ((1.0 - dp_mask) * action_batch)

            log_pi, _ = self.actor.get_log_prob(state_batch, actions, with_grad=False)
            
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha().exp().detach()

    def update_critic(self) -> None:
        for _ in range(self.n_critic_updates):
            batch = self.buffer.sample()
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
        self.log_alpha.load_state_dict(torch.load(log_alpha_path, map_location=device))

        alpha_opt_path = path / "alpha_opt"
        self.alpha_optimizer.load_state_dict(torch.load(alpha_opt_path, map_location=device))

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)