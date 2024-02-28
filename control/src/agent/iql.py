import os
import torch
from torch.nn.utils import clip_grad_norm_

from src.agent.base import BaseAC
from src.network.factory import init_optimizer, init_custom_network
import src.network.torch_utils as torch_utils


class IQL(BaseAC):
    def __init__(self, cfg):
        super(IQL, self).__init__(cfg)

        self.clip_grad_param = 100
        self.temperature = cfg.tau  # 3
        self.expectile = cfg.rho  # torch.FloatTensor([0.8]).to(device)

        self.value_net = init_custom_network("FC", cfg.device, self.state_dim, cfg.hidden_critic, 1,
                                             cfg.activation, "None", cfg.layer_init_critic, cfg.layer_norm)
        self.value_optimizer = init_optimizer(cfg.optimizer, list(self.value_net.parameters()), cfg.lr_critic)

    def compute_loss_pi(self, data):
        states, actions = data['obs'], data['act']
        with torch.no_grad():
            v = self.value_net(states)
        min_Q, _, _ = self.get_q_value_target(states, actions)
        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device))#.squeeze(-1)
        log_probs = self.ac.pi.get_logprob(states, actions)
        actor_loss = -(exp_a * log_probs).mean()
        print("pi loss", v.size(), min_Q.size(), exp_a.size(), log_probs.size())
        return actor_loss

    def compute_loss_value(self, data):
        states, actions = data['obs'], data['act']
        min_Q, _, _ = self.get_q_value_target(states, actions)

        value = self.value_net(states)
        value_loss = torch_utils.expectile_loss(min_Q - value, self.expectile).mean()
        print("value loss", min_Q.size(), value.size(), torch_utils.expectile_loss(min_Q - value, self.expectile).size())
        return value_loss

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data[
            'done']
        with torch.no_grad():
            next_v = self.value_net(next_states)#.squeeze(-1)
            target = rewards + (self.gamma * (1 - dones) * next_v)
        _, q_ens = self.get_q_value(states, actions, with_grad=True)
        q_loss = self.ensemble_mse(target, q_ens)
        print("q loss", next_v.size(), target.size(), q_ens[0].size())
        return q_loss

    def inner_update(self, trunc=False):
        data = self.get_data()

        self.value_optimizer.zero_grad()
        loss_vs = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q = self.compute_loss_q(data)
        self.critic_optimizer.zero_grad()
        self.ensemble_critic_loss_backward(loss_q) #loss_q.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        self.critic_optimizer.step()

        loss_pi = self.compute_loss_pi(data)
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        return {}

    def save(self):
        super(IQL, self).save()
        parameters_dir = self.parameters_dir

        path = os.path.join(parameters_dir, "value_net")
        torch.save(self.value_net.state_dict(), path)

        path = os.path.join(parameters_dir, "value_opt")
        torch.save(self.value_optimizer.state_dict(), path)
