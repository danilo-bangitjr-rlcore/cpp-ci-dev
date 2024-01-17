import numpy as np
import os
import torch
from src.agent.base import BaseAC
import src.network.torch_utils as torch_utils

class SAC(BaseAC):
    def __init__(self, cfg):
        super(SAC, self).__init__(cfg)
        self.automatic_entropy_tuning = cfg.tau == -1
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(self.action_dim).item()
            self.log_alpha = torch_utils.Float(self.device, 0.0)
        else:
            self.log_alpha = torch_utils.Float(self.device, np.log(cfg.tau))
        self.alpha = self.log_alpha().exp().detach()
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.cfg.lr_actor)

        if cfg.load_path != "":
            self.load(cfg.load_path, cfg.load_checkpoint)

    def compute_loss_q(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        next_state_action, next_state_log_pi, _ = self.get_policy(next_state_batch, with_grad=False)
        q_pi_targ, _ = self.get_q_value_target(next_state_batch, self.action_normalizer(next_state_action))
        q_pi_targ -= self.alpha * next_state_log_pi
        # with torch.no_grad():
        #     q_pi_targ = self.v_baseline(next_state_batch)#.squeeze(-1)    # v is trained with entropy
        next_q_value = reward_batch + mask_batch * self.gamma * q_pi_targ
        q, _ = self.get_q_value(state_batch, self.action_normalizer(action_batch), with_grad=True)
        critic_loss = torch.nn.functional.mse_loss(q, next_q_value)
        return critic_loss

    def compute_loss_pi(self, state_batch, action_batch):
        pi, log_pi, _ = self.get_policy(state_batch, with_grad=True)  # self.ac.pi(state_batch)
        min_qf_pi, _ = self.get_q_value(state_batch, self.action_normalizer(pi), with_grad=True)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        return policy_loss, log_pi

    def update_entropy(self, state_batch):
        _, log_pi, _ = self.get_policy(state_batch, with_grad=False)
        alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha().exp().detach()
        return

    # def compute_loss_value(self, data):
    #     """L_{\phi}, learn z for state value, v = tau log z"""
    #     states = data['obs']
    #     v_phi = self.v_baseline(states)#.squeeze(-1)
    #     actions, log_probs, _ = self.get_policy(states, with_grad=False)  # self.ac.pi(states)
    #     min_Q, _ = self.get_q_value_target(states, self.action_normalizer(actions))
    #     target = min_Q - self.alpha * log_probs
    #     value_loss = (0.5 * (v_phi - target) ** 2).mean()
    #     return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done']
    
        # # if self.random_ensemble:
        # self.v_optimizer.zero_grad()
        # loss_vs, v_info, logp_info = self.compute_loss_value(data)
        # loss_vs.backward()
        # self.v_optimizer.step()
    
        self.critic_optimizer.zero_grad()
        qf_loss = self.compute_loss_q(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        qf_loss.backward()
        self.critic_optimizer.step()
    
        policy_loss, log_pi = self.compute_loss_pi(state_batch, action_batch)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
    
        if self.automatic_entropy_tuning:
            self.update_entropy(state_batch)
    
    def save(self):
        super(SAC, self).save()
        parameters_dir = self.parameters_dir
        path = os.path.join(parameters_dir, "log_alpha")
        torch.save(self.log_alpha.state_dict(), path)
    
        path = os.path.join(parameters_dir, "alpha_opt")
        torch.save(self.alpha_optimizer.state_dict(), path)

    def load(self, parameters_dir, checkpoint=False):
        pth = os.path.join(parameters_dir, 'log_alpha')
        self.log_alpha.load_state_dict(torch.load(pth, map_location=self.device))
        self.logger.info("Load log(alpha) from {}".format(pth))
    
        if checkpoint:
            pth = os.path.join(parameters_dir, 'alpha_opt')
            self.alpha_optimizer.load_state_dict(torch.load(pth, map_location=self.device))
            self.logger.info("Load alpha optimizer from {}".format(pth))
            