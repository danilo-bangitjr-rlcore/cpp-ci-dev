import os
import torch
import pickle as pkl
import torch.nn as nn
import numpy as np
from src.network.networks import FC, BetaPolicy
from src.agent.base import BaseAC
import src.network.torch_utils as torch_utils

class Reinforce(BaseAC):
    def __init__(self, cfg):
        super(Reinforce, self).__init__(cfg=cfg)
        self.v_baseline = FC(self.device, self.state_dim, cfg.hidden_critic, 1)
        self.v_optimizer = torch.optim.RMSprop(list(self.v_baseline.parameters()), cfg.lr_v)

        self.ep_states = [self.observation]
        self.ep_actions = []
        self.ep_rewards = []

    def step(self):
        action, _, pi_info = self.get_policy(torch_utils.tensor(self.observation.reshape((1, -1)), self.device), with_grad=False, debug=self.cfg.debug)
        action = torch_utils.to_np(action)[0]
        next_observation, reward, terminated, trunc, env_info = self.env.step(action)
        
        i_log = {
            "agent_info": pi_info,
            "env_info": env_info
        }
        self.info_log.append(i_log)

        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_states.append(next_observation)

        reset, truncate = self.update_stats(reward, terminated, trunc)
        
        if reset:
            self.update(truncate)
            next_observation, info = self.env.reset()
            self.ep_states = [next_observation]
            self.ep_actions = []
            self.ep_rewards = []

        self.observation = next_observation

        return

    def inner_update(self, trunc=False):
        ep_t = len(self.ep_states) - 1
        G = 0.0

        # If the episode is truncated, returns bootstrap the final state
        if trunc:
            v_boot = self.get_v_value(torch_utils.tensor(self.state_normalizer(self.ep_states[ep_t]).reshape((1, -1)), self.device), with_grad=False)
            G = v_boot

        returns = np.zeros(ep_t)

        ep_t -= 1

        for t in range(ep_t, -1, -1):
            G = self.ep_rewards[t] + self.gamma * G
            returns[t] = G

        returns = torch_utils.tensor(returns, self.device)

        self.ep_states = np.asarray(self.ep_states[:-1])
        self.ep_states = torch_utils.tensor(self.state_normalizer(self.ep_states), self.device)
        self.ep_actions = np.asarray(self.ep_actions)
        self.ep_actions = torch_utils.tensor(self.ep_actions, self.device)

        v_base = self.get_v_value(self.ep_states, with_grad=True)

        with torch.no_grad():
            delta = returns - v_base

        # Update Actor
        log_prob, _ = self.actor.log_prob(self.ep_states, self.ep_actions)
        log_prob = log_prob.view(-1,1)

        actor_loss = torch.mean(-log_prob * delta)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Baseline
        v_base_loss = nn.functional.mse_loss(v_base, returns)
        
        self.v_optimizer.zero_grad()
        v_base_loss.backward()
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
