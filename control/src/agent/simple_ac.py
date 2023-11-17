import os
import torch
import pickle as pkl
import torch.nn as nn
from src.network.networks import FC, BetaPolicy
from src.agent.base import BaseAC

class SimpleAC(BaseAC):
    def __init__(self, cfg):
        super(SimpleAC, self).__init__(cfg=cfg)
        self.tau = self.cfg.tau
        self.v_baseline = FC(self.device, self.state_dim, cfg.hidden_units, 1)
        self.v_optimizer = torch.optim.RMSprop(list(self.v_baseline.parameters()), cfg.lr_v)

    def inner_update(self):
        batch = self.get_data()
        
        log_prob, dist = self.actor.log_prob(batch['obs'], batch['act'])
        
        v = self.get_v_value(batch['obs'], with_grad=True)
        vp = self.get_v_value(batch['obs2'], with_grad=False)
        targ = batch['reward'] + self.gamma * (1.0 - batch['done']) * vp
        ent = dist.entropy().unsqueeze(-1)
        loss_actor = -(self.tau * ent + log_prob * (targ - v.detach())).mean()
        loss_critic = nn.functional.mse_loss(v, targ)
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        self.v_optimizer.zero_grad()
        loss_critic.backward()
        self.v_optimizer.step()

    def save(self):
        parameters_dir = self.parameters_dir
    
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.actor.state_dict(), path)
    
        path = os.path.join(parameters_dir, "actor_opt")
        torch.save(self.actor_optimizer.state_dict(), path)
    
        path = os.path.join(parameters_dir, "v_baseline_net")
        torch.save(self.v_baseline.state_dict(), path)
    
        path = os.path.join(parameters_dir, "v_baseline_opt")
        torch.save(self.v_optimizer.state_dict(), path)
        
        path = os.path.join(parameters_dir, "buffer.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.buffer, f)
