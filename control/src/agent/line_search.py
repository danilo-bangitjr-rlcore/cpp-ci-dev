import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.network.factory import init_policy_network, init_critic_network, init_optimizer, init_custom_network
import src.network.torch_utils as torch_utils
from src.component.line_search_opt import LineSearchOpt
from src.component.exploration import RndNetworkExplore


class LineSearchGAC(GreedyAC):
    def __init__(self, cfg, average_entropy=True):
        super(LineSearchGAC, self).__init__(cfg, average_entropy)

        # Leave positions for backup the networks
        if self.discrete_control:
            self.top_action = 1
            self.actor_copy = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                                  cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                  cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic_copy = init_critic_network(cfg.critic, cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                   cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.random_policy = init_policy_network("UniformRandomDisc", cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                     cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                     cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm
                                                     )
        else:
            self.actor_copy = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                                  cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                  cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
            self.critic_copy = init_critic_network(cfg.critic, cfg.device, self.state_dim + self.action_dim, cfg.hidden_critic, 1,
                                                   cfg.activation, cfg.layer_init_critic, cfg.layer_norm, cfg.critic_ensemble)
            self.random_policy = init_policy_network("UniformRandomCont", cfg.device, self.state_dim, cfg.hidden_critic, self.action_dim,
                                                     cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                                     cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm
                                                     )

        self.sampler_copy = init_policy_network(cfg.actor, cfg.device, self.state_dim, cfg.hidden_actor, self.action_dim,
                                           cfg.beta_parameter_bias, cfg.beta_parameter_bound, cfg.activation,
                                           cfg.head_activation, cfg.layer_init_actor, cfg.layer_norm)
        self.lr_sampler = cfg.lr_actor

        self.explore_network = RndNetworkExplore(cfg.device, self.state_dim, self.action_dim, cfg.hidden_critic, cfg.activation)
        self.fbonus0, self.fbonus1, self.fbonus0_copy, self.fbonus1_copy = self.explore_network.get_networks()

        self.lr_explore = cfg.lr_critic
        self.explore_scaler = cfg.exploration

        # Ensure not using the wrong optimizer
        del self.actor_optimizer
        del self.critic_optimizer
        del self.sampler_optim

        self.actor_linesearch = LineSearchOpt([self.actor], [self.actor_copy])
        self.critic_linesearch = LineSearchOpt([self.critic], [self.critic_copy])
        self.sampler_linesearch = LineSearchOpt([self.sampler], [self.sampler_copy])
        self.explore_linesearch = LineSearchOpt([self.fbonus0, self.fbonus1], [self.fbonus0_copy, self.fbonus1_copy])
        self.last_explore_bonus = None

    def explore_bonus_update(self, state, action, reward, next_state, next_action, mask,
                         eval_state, eval_action, eval_reward, eval_next_state, eval_mask):
        random_next_action, _, _ = self.random_policy(state)
        loss0, loss1 = self.explore_network.explore_bonus_loss(state, action, reward, next_state, random_next_action,
                                                               mask, self.gamma)
        self.explore_linesearch.backtrack(error_evaluation_fn=self.eval_error_bonus,
                                          error_eval_input=[eval_state, eval_action],
                                          network_lst=[self.fbonus0, self.fbonus1],
                                          loss_lst=[loss0, loss1])

    def explore_bonus_eval(self, state, action):
        return self.explore_network.explore_bonus_eval(state, action)

    def eval_error_critic(self, args):
        state_batch, action_batch, reward_batch, mask_batch, next_q = args
        q, _ = self.get_q_value(state_batch, action_batch, with_grad=False)
        target = reward_batch + mask_batch * self.gamma * next_q
        error = nn.functional.mse_loss(q.detach(), target.detach())
        return error

    def eval_error_actor(self, args):
        state_batch, action_batch, network = args
        with torch.no_grad():
            logp, _ = network.log_prob(state_batch, action_batch)
        return -logp.mean().detach()

    def eval_error_bonus(self, args):
        state, action = args
        return self.explore_network.explore_bonus_eval(state, action).mean()

    def critic_update(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                         eval_state, eval_action, eval_reward, eval_next_state, eval_mask):
        eval_next_action, _, _ = self.get_policy(eval_next_state, with_grad=False)
        eval_next_q, _ = self.get_q_value_target(eval_next_state, eval_next_action)
        q_loss, next_action = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.critic_linesearch.backtrack(error_evaluation_fn=self.eval_error_critic,
                                         error_eval_input=[eval_state, eval_action, eval_reward, eval_mask, eval_next_q],
                                         network_lst=[self.critic],
                                         loss_lst=[q_loss])
        return next_action

    def get_action_with_top_value(self, eval_state, eval_sample_actions, sorted_eval_q, count_top):
        eval_best = sorted_eval_q[:, : count_top]
        eval_best = eval_best.repeat_interleave(self.gac_a_dim, -1)
        eval_sample_actions = eval_sample_actions.reshape(eval_state.shape[0], self.num_samples, self.gac_a_dim)
        eval_best_action = torch.gather(eval_sample_actions, 1, eval_best)
        eval_best_action = torch.reshape(eval_best_action, (-1, self.gac_a_dim))
        eval_stacked_s = eval_state.repeat_interleave(count_top, dim=0)
        return eval_stacked_s, eval_best_action

    def get_best_action_proposal(self, eval_state):
        eval_size = eval_state.shape[0]
        eval_rept_states = eval_state.repeat_interleave(self.num_samples, dim=0)
        with torch.no_grad():
            eval_sample_actions, _, _ = self.sampler(eval_rept_states)
        eval_q, _ = self.get_q_value(eval_rept_states, eval_sample_actions, with_grad=False)
        eval_q = eval_q.reshape(eval_size, self.num_samples, 1)
        sorted_eval_q = torch.argsort(eval_q, dim=1, descending=True)
        eval_stacked_s, eval_best_action = self.get_action_with_top_value(eval_state, eval_sample_actions, sorted_eval_q, self.top_action)
        return eval_stacked_s, eval_best_action, eval_sample_actions, sorted_eval_q

    def sort_q_value(self, repeated_states, sample_actions, batch_size):
        # Add the exploration bonus
        q_values, _ = self.get_q_value(repeated_states, sample_actions, with_grad=False)
        exp_b = self.explore_bonus_eval(repeated_states, sample_actions)
        # print("sortqvalue")
        # print(q_values.mean(), q_values.std(), q_values.min(), q_values.max())
        # print(exp_b.mean(), exp_b.std(), exp_b.min(), exp_b.max())
        # print("---")
        self.last_explore_bonus = [exp_b.mean(), exp_b.min(), exp_b.max()]
        q_values += self.explore_scaler * exp_b

        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q

    def actor_update(self, state_batch, eval_state):
        pi_loss, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, logp = self.actor_loss(state_batch)
        eval_stacked_s, eval_best_action, eval_sample_actions, sorted_eval_q = self.get_best_action_proposal(eval_state)
        self.actor_linesearch.backtrack(error_evaluation_fn=self.eval_error_actor,
                                        error_eval_input=[eval_stacked_s, eval_best_action, self.actor],
                                        network_lst=[self.actor],
                                        loss_lst=[pi_loss])
        return (sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                eval_sample_actions, sorted_eval_q)

    def sampler_update(self, sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                          eval_state, eval_sample_actions, sorted_eval_q):
        sampler_loss = self.proposal_loss(sample_actions, repeated_states,
                                          stacked_s_batch, best_actions, sorted_q, state_batch)
        eval_stacked_s, eval_best_action = self.get_action_with_top_value(eval_state, eval_sample_actions, sorted_eval_q,
                                                                          self.top_action_proposal)
        self.sampler_linesearch.backtrack(error_evaluation_fn=self.eval_error_actor, # use the same error evaluation function as the actor
                                          error_eval_input=[eval_stacked_s, eval_best_action, self.sampler],
                                          network_lst=[self.sampler],
                                          loss_lst=[sampler_loss])
        return

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (data['obs'], data['act'], data['reward'],
                                                                                 data['obs2'], 1. - data['done'])

        # eval_data = self.get_all_data() # get batch
        eval_data = self.get_data() # get minibatch
        eval_state, eval_action, eval_reward, eval_next_state, eval_mask = (eval_data['obs'], eval_data['act'], eval_data['reward'],
                                                                            eval_data['obs2'], 1. - eval_data['done'])

        # critic update
        next_action = self.critic_update(state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                                            eval_state, eval_action, eval_reward, eval_next_state, eval_mask)

        # uncertainty update
        self.explore_bonus_update(state_batch, action_batch, reward_batch, next_state_batch, next_action, mask_batch,
                                  eval_state, eval_action, eval_reward, eval_next_state, eval_mask)

        # actor update
        sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch, eval_sample_actions, sorted_eval_q = \
            self.actor_update(state_batch, eval_state)

        # proposal update
        self.sampler_update(sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                               eval_state, eval_sample_actions, sorted_eval_q)

    def agent_debug_info(self, observation_tensor, action_tensor, pi_info, env_info):
        i_log = super(LineSearchGAC, self).agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        i_log["LS_actor"] = self.actor_linesearch.debug_info()
        i_log["LS_critic"] = self.critic_linesearch.debug_info()
        i_log["LS_sampler"] = self.sampler_linesearch.debug_info()
        i_log["LS_explore"] = self.explore_linesearch.debug_info()
        i_log["explore_bonus"] = self.last_explore_bonus
        return i_log

    def save(self):
        parameters_dir = self.parameters_dir

        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.critic.state_dict(), path)

        path = os.path.join(parameters_dir, "critic_target")
        torch.save(self.critic_target.state_dict(), path)

        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.actor.state_dict(), path)

        path = os.path.join(parameters_dir, "sampler_net")
        torch.save(self.sampler.state_dict(), path)

        path = os.path.join(parameters_dir, "buffer.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.buffer, f)

