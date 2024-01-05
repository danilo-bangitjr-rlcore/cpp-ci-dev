import os
import torch
import pickle as pkl
import torch.nn as nn
import numpy as np
from src.network.networks import FC, BetaPolicy
from src.agent.base import BaseAC
import src.network.torch_utils as torch_utils

# https://tor-lattimore.com/downloads/book/book.pdf chapter 6
class ExploreThenCommit(BaseAC):
    def __init__(self, cfg):
        super(ExploreThenCommit, self).__init__(cfg=cfg)
        self.actions, self.shape = self.env.get_action_samples(n=cfg.actions_per_dim)
        self.num_actions = np.prod(self.shape)
        self.exploration_trials = self.num_actions * cfg.min_trials
        self.counts = np.zeros(self.num_actions)
        self.reward_sum = np.zeros(self.num_actions)
        self.best_value = -np.inf
        self.best_action = None

    def choose_action(self):
        if self.num_episodes < self.exploration_trials:
            return self.actions[self.num_episodes % self.num_actions]
        else:
            return self.best_action

    def update_action_values(self, reward):
        if self.num_episodes < self.exploration_trials:
            action_num = self.num_episodes % self.num_actions
            self.counts[action_num] += 1
            self.reward_sum[action_num] += reward
            curr_value = self.reward_sum[action_num] / self.counts[action_num]
            if curr_value > self.best_value:
                self.best_value = curr_value
                self.best_action = self.actions[action_num]

    def step(self):
        action = self.choose_action()
        next_observation, reward, terminated, trunc, env_info = self.env.step(action)
        
        i_log = {
            "agent_info": None,
            "env_info": env_info
        }
        self.info_log.append(i_log)

        self.update_action_values(reward)

        reset, truncate = self.update_stats(reward, terminated, trunc)
        
        if reset:
            next_observation, info = self.env.reset()

        self.observation = next_observation

        return


    def save(self):
        pass
