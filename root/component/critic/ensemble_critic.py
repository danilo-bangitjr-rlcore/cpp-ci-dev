from base_critic import BaseQ, BaseV
from root.component.optimizers.factory import init_optimizer
from root.component.network.factory import init_critic_network
import torch

class EnsembleQCritic(BaseQ):
    def __init__(self, cfg, state_dim, action_dim):
        self.model = init_critic_network(cfg.network, state_dim, action_dim)
        self.target = init_critic_network(cfg.network, state_dim, action_dim)
        self.optimizer = init_optimizer(cfg.optimizer) # TODO
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_q(self, states, actions, with_grad=False):
        state_actions = torch.concat((states, actions), dim=1)
        if with_grad:
            q, qs = self.model(state_actions)
        else:
            with torch.no_grad():
                q, qs = self.model(state_actions)
        return q, qs

    def get_q_target(self, states, actions):
        state_actions = torch.concat((states, actions), dim=1)
        with torch.no_grad():
            q, qs = self.target(state_actions)
        return q, qs

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


class EnsembleVCritic(BaseV):
    def __init__(self, cfg, state_dim):
        self.model = init_critic_network(cfg.network, state_dim, output_dim=1)
        self.target = init_critic_network(cfg.network, state_dim, output_dim=1)
        self.optimizer = init_optimizer(cfg.optimizer) # TODO
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_v(self, states, with_grad=False):
        if with_grad:
            v, vs = self.model(states)
        else:
            with torch.no_grad():
                v, vs = self.model(states)
        return v, vs

    def get_v_target(self, states):
        with torch.no_grad():
            v, vs = self.target(states)
        return v, vs

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)