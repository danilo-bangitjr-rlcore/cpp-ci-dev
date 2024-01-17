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
        self.critic_update = 'on_policy'
        self.etc_buffer_prefill = cfg.etc_buffer_prefill
        self.learning_start = cfg.etc_learning_start
       


    def choose_action(self):
        if self.num_episodes < self.exploration_trials:
            return self.actions[self.num_episodes % self.num_actions]
        else:
            return self.best_action
        
    def choose_next_action(self):
        selection_time = self.num_episodes + 1 # the time to select actions
        if selection_time < self.exploration_trials:
            return self.actions[selection_time % self.num_actions]
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
        self.update_action_values(reward)

        reset, truncate = self.update_stats(reward, terminated, trunc)
        
        if reset:
            next_observation, info = self.env.reset()
            
        # fill the buffer, only if exploration period is still going on
        if self.num_episodes <= self.etc_buffer_prefill: # only add new transitions while less that 
            self.buffer.feed([self.observation, action, reward, next_observation, int(terminated), int(truncate)])
        
        if self.num_episodes >= self.learning_start: # when learning starts
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                    data['obs2'], 1 - data['done']

            # select which action to use for critic update. 
            # NOTE: this doesn't matter for bandits, but could be a future consideration if we want to use ETC
            # in MDPs
            if self.critic_update == 'on_policy': # chooses the next action according to ETC
                next_action = self.choose_next_action()
            elif self.critic_update == 'greedy': # chooses the current "best action". Sort of Q-learning
                next_action = self.best_action
            elif self.critic_update == 'repeat': # chooses same action again
                next_action = action
            
            # expand to appropriate number of dimensions then cast to tensor
            next_action = np.ones((self.batch_size, next_action.shape[0])) * next_action 
            next_action = torch_utils.tensor(next_action, self.device)
            
            next_q, _ = self.get_q_value_target(next_state_batch, self.action_normalizer(next_action))
            target = reward_batch  + mask_batch * self.gamma * next_q
            
            q_value, _ = self.get_q_value(state_batch, self.action_normalizer(action_batch), with_grad=True)
            q_loss = torch.nn.functional.mse_loss(target, q_value)
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            
        # debug logs
        action = torch_utils.tensor([action], self.device)
        
        observation_tensor = torch_utils.tensor(self.observation.reshape((1, -1)), self.device)
        _ , _, pi_info = self.get_policy(observation_tensor,
                                                    with_grad=False, debug=self.cfg.debug)
        i_log = self.agent_debug_info(observation_tensor, action, pi_info, env_info)
        
        self.info_log.append(i_log)
        
        # render
        if self.cfg.render:
            self.render(np.array(env_info['interval_log']), i_log['critic_info']['Q-function'], i_log["action_visits"])
        else:
            env_info.pop('interval_log', None)
            i_log['critic_info'].pop('Q-function', None)


        self.observation = next_observation

        return
    

    def save(self):
        pass
