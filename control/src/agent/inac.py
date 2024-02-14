import os
import numpy as np
import torch
from src.agent.base import BaseAC
import src.network.torch_utils as torch_utils
from src.network.factory import init_policy_network, init_optimizer, init_custom_network


class InAC(BaseAC):
    def __init__(self, cfg):
        super(InAC, self).__init__(cfg)
        self.beh_pi = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                          cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                          cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
        self.beh_pi_optimizer = init_optimizer(cfg.optimizer, list(self.beh_pi.parameters()), cfg.lr_actor)

        self.value_net = init_custom_network("FC", cfg.device, self.state_dim, cfg.hidden_critic, 1,
                                             cfg.activation, "None", cfg.layer_init_critic, cfg.layer_norm)
        self.value_optimizer = init_optimizer(cfg.optimizer, list(self.value_net.parameters()), cfg.lr_critic)

        if cfg.tau == -1:
            self.log_temperature = torch_utils.Float(cfg.device, 0.0)
            self.temperature_optimizer = torch.optim.Adam(
                self.log_temperature.parameters(),
                lr=cfg.learning_rate,
            )
            self.target_entropy = -np.prod(self.cfg.action_dim).item()
        else:
            self.log_temperature = torch_utils.Float(cfg.device, np.log(cfg.tau))
        self.tau = self.log_temperature().exp().detach()
        self.eps = 1e-8
        self.exp_threshold = 10000
        return

    def get_state_value(self, state):
        with torch.no_grad():
            value = self.value_net(state).squeeze(-1)
        return value

    def compute_loss_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs, _ = self.beh_pi.log_prob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs

    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states)
        actions, log_probs, _ = self.get_policy(states, with_grad=False)
        min_Q, _ = self.get_q_value_target(states, actions)
        target = min_Q - self.tau * log_probs
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = (data['obs'], data['act'], data['reward'],
                                                        data['obs2'], data['done'])
        minq, _ = self.get_q_value(states, actions, with_grad=True)
        a2, logp_a2, _ = self.get_policy(next_states, with_grad=False)  # self.ac.pi(op)
        q_pi_targ, _ = self.get_q_value_target(next_states, a2)
        q_pi_targ -= self.tau * logp_a2
        target = rewards + self.gamma * (1-dones) * q_pi_targ
        loss = torch.nn.functional.mse_loss(minq, target)
        return loss

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']

        log_probs, _ = self.actor.log_prob(states, actions)
        min_Q, _ = self.get_q_value(states, actions, with_grad=False)
        min_Q = min_Q.squeeze(-1)
        with torch.no_grad():
            value = self.get_state_value(states)
            beh_log_prob, _ = self.beh_pi.log_prob(states, actions)
        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""

    def update_beta(self, data):
        loss_beh_pi, logp = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()
        return loss_beh_pi

    def update_entropy(self, state_batch):
        _, log_pi, _ = self.get_policy(state_batch, with_grad=False)
        alpha_loss = -(self.log_temperature() * (log_pi + self.target_entropy).detach()).mean()
        self.tau = self.log_temperature().exp().detach()
        self.temperature_optimizer.zero_grad()
        alpha_loss.backward()
        self.temperature_optimizer.step()
        return

    def inner_update(self, trunc=False):
        data = self.get_data()

        loss_beta = self.update_beta(data).item()

        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q = self.compute_loss_q(data)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        if self.cfg.tau == -1:
            self.update_entropy(data['obs'])

        return {}

    def save(self):
        parameters_dir = self.parameters_dir

        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)

        path = os.path.join(parameters_dir, "actor_opt")
        torch.save(self.pi_optimizer.state_dict(), path)

        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)

        path = os.path.join(parameters_dir, "critic_net_target")
        torch.save(self.ac_targ.q1q2.state_dict(), path)

        path = os.path.join(parameters_dir, "critic_opt")
        copt_dic = {}
        for i, opt in enumerate(self.q_optimizer):
            copt_dic[i] = opt.state_dict()
        torch.save(copt_dic, path)

        path = os.path.join(parameters_dir, "value_net")
        torch.save(self.value_net.state_dict(), path)

        path = os.path.join(parameters_dir, "value_opt")
        torch.save(self.value_optimizer.state_dict(), path)

        path = os.path.join(parameters_dir, "log_alpha")
        torch.save(self.log_temperature, path)
        log_alpha_loc = torch.load(path)  # something weird going on here
        torch.save(log_alpha_loc.state_dict(), path)

        if self.cfg.tau == -1:
            path = os.path.join(parameters_dir, "alpha_opt")
            torch.save(self.temperature_optimizer.state_dict(), path)

        path = os.path.join(parameters_dir, "beh_net")
        torch.save(self.beh_pi.state_dict(), path)

        path = os.path.join(parameters_dir, "beh_opt")
        torch.save(self.beh_pi_optimizer.state_dict(), path)

