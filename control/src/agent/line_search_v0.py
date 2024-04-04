import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.agent.greedy_ac import GreedyAC, GreedyACDiscrete
from src.network.factory import init_policy_network, init_critic_network, init_optimizer, init_custom_network
import src.network.torch_utils as torch_utils
from src.network.torch_utils import clone_model_0to1, clone_gradient, move_gradient_to_network
from src.component.normalizer import init_normalizer


class LineSearchAgent(GreedyAC):

    """
    Based on GAC

    [done] remove resetting; increase the backtracking length; change to sgd
    [done] Use minibatch for update; use batch/another minibatch for test; increase the threshold
    [done] Add exploration
      *    Change to value based. Qt-opt?
    """
    def __init__(self, cfg, average_entropy=True):
        super(LineSearchAgent, self).__init__(cfg, average_entropy)

        # Leave positions for backup the actor_network
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

        self.actor_opt_copy = init_optimizer(cfg.optimizer, list(self.actor_copy.parameters()), cfg.lr_actor)
        self.sampler_opt_copy = init_optimizer(cfg.optimizer, list(self.sampler_copy.parameters()), self.lr_sampler)
        self.critic_opt_copy = init_optimizer(cfg.optimizer, list(self.critic_copy.parameters()), cfg.lr_critic)


        self.actor_lr_weight = 1  # always start with 1, tend to use a large learning rate initialization (cfg.lr_critic)
        self.actor_lr_weight_copy = 1
        self.sampler_lr_weight = 1
        self.sampler_lr_weight_copy = 1
        self.critic_lr_weight = 1
        self.critic_lr_weight_copy = 1
        self.explore_lr_weight = 1
        self.explore_lr_weight_copy = 1
        self.last_actor_scaler = None
        self.last_sampler_scaler = None
        self.last_critic_scaler = None
        self.last_explore_scaler = None
        self.last_explore_bonus = None
        self.separated_testset = True


        self.actor_lr_start = self.cfg.lr_actor
        self.critic_lr_start = self.cfg.lr_critic

        # Networks for uncertainty measure
        self.ftrue0 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.ftrue1 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus0 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus1 = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus0_copy = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus1_copy = init_custom_network("RndLinearUncertainty", cfg.device, self.state_dim+self.action_dim,
                                          cfg.hidden_critic, self.state_dim+self.action_dim, cfg.activation, "None",
                                          "Xavier/1", layer_norm=False)

        # Ensure the nonlinear net between learning net and target net are the same
        clone_model_0to1(self.ftrue0.random_network, self.fbonus0.random_network)
        clone_model_0to1(self.ftrue1.random_network, self.fbonus1.random_network)
        clone_model_0to1(self.fbonus0, self.fbonus0_copy)
        clone_model_0to1(self.fbonus1, self.fbonus1_copy)

        # actor_optimizer for learning net only
        self.lr_explore = cfg.lr_critic
        self.bonus_opt_0 = init_optimizer(cfg.optimizer, list(self.fbonus0.parameters()), self.lr_explore)
        self.bonus_opt_1 = init_optimizer(cfg.optimizer, list(self.fbonus1.parameters()), self.lr_explore)
        self.bonus_opt_0_copy = init_optimizer(cfg.optimizer, list(self.fbonus0.parameters()), self.lr_explore)
        self.bonus_opt_1_copy = init_optimizer(cfg.optimizer, list(self.fbonus1.parameters()), self.lr_explore)

        # TODO: These should go to variable in main function later
        self.max_backtracking = 30
        self.reducing_rate = 0.5
        self.increasing_rate = 1.1 # Not sure if this is going to be used
        self.critic_lr_lower_bound = 1e-06
        self.actor_lr_lower_bound = 1e-06
        self.error_threshold = 1e-4
        self.explore_scaler = cfg.exploration

        # assert self.cfg.batch_size >= 256
        assert self.max_backtracking > 0
        self.random_prefill()


    def random_prefill(self):
        pth = self.cfg.parameters_path + "/prefill_{}.pkl".format(self.cfg.etc_buffer_prefill)
        if not os.path.isfile(pth):
            '''using etc's parameter here'''
            reset = True
            for t in range(self.cfg.etc_buffer_prefill):
                if t % 1000 == 0:
                    print("Prefill t={}".format(t))
                if reset:
                    observation, info = self.env_reset()
                observation_tensor = torch_utils.tensor(observation.reshape((1, -1)), self.device)
                with torch.no_grad():
                    action, _, _ = self.random_policy(observation_tensor, False)
                    action = torch_utils.to_np(action)
                next_observation, reward, terminated, trunc, env_info = self.env_step(action)
                reset = terminated or trunc
                self.buffer.feed([observation, action[0], reward, next_observation, int(terminated), int(trunc)])
                observation = next_observation
            with open(pth, "wb") as f:
                pkl.dump(self.buffer, f)
        else:
            print("Load prefilled data from", pth)
            with open(pth, "rb") as f:
                self.buffer = pkl.load(f)
            self.buffer.batch_size = self.batch_size

        # # For debugging
        # _, ax = plt.subplots(1, 1, figsize=(4, 4))
        # data = self.buffer.get_all_data()
        # act = np.array([i[1] for i in data])
        # rwd = np.array([i[2] for i in data])
        # ax.scatter(act[:, 0], act[:, 1], c=rwd, s=5)
        # ax.invert_yaxis()
        # plt.show()

    def explore_bonus_update(self, state, action, reward, next_state, next_action, mask,
                         eval_state, eval_action, eval_reward, eval_next_state, eval_mask):
        before_error = self.explore_bonus_eval(eval_state, eval_action).mean()

        in_ = torch.concat((state, action), dim=1)
        in_p1 = torch.concat((next_state, action), dim=1)
        with torch.no_grad():
            true0_t, _ = self.ftrue0(in_)
            true1_t, _ = self.ftrue1(in_)
            true0_tp1, _ = self.ftrue0(in_p1)
            true1_tp1, _ = self.ftrue1(in_p1)
        reward0 = true0_t - mask * self.gamma * true0_tp1
        reward1 = true1_t - mask * self.gamma * true1_tp1

        pred0, _ = self.fbonus0(in_)
        pred1, _ = self.fbonus1(in_)
        with torch.no_grad():
            target0 = reward0.detach() + mask * self.gamma * self.fbonus0(in_p1)[0] # true0_t #
            target1 = reward1.detach() + mask * self.gamma * self.fbonus1(in_p1)[0] # true1_t #

        # TODO: Include the reward and next state in training, for the stochasity (how?)
        loss0 = nn.functional.mse_loss(pred0, target0)
        loss1 = nn.functional.mse_loss(pred1, target1)

        self.bonus_opt_0.zero_grad()
        loss0.backward()
        grad_rec_exp0 = clone_gradient(self.fbonus0)
        self.bonus_opt_1.zero_grad()
        loss1.backward()
        grad_rec_exp1 = clone_gradient(self.fbonus1)
        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                self.bonus_opt_0.zero_grad()
                self.bonus_opt_1.zero_grad()
                move_gradient_to_network(self.fbonus0, grad_rec_exp0, self.explore_lr_weight)
                move_gradient_to_network(self.fbonus1, grad_rec_exp1, self.explore_lr_weight)
            self.bonus_opt_0.step()
            self.bonus_opt_1.step()

            after_error = self.explore_bonus_eval(eval_state, eval_action).mean()

            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.explore_lr_weight *= 0.5
                self.undo_update_explore()
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                self.lr_explore = max(self.cfg.lr_explore * 0.5, self.critic_lr_lower_bound)
                self.bonus_opt_0 = init_optimizer(self.cfg.optimizer, list(self.fbonus0.parameters()), self.lr_explore)
                self.bonus_opt_1 = init_optimizer(self.cfg.optimizer, list(self.fbonus1.parameters()), self.lr_explore)
                break
            else:
                break
        self.last_explore_scaler = self.explore_lr_weight
        self.explore_lr_weight = self.explore_lr_weight_copy


    def explore_bonus_eval(self, state, action):
        in_ = torch.concat((state, action), dim=1)
        with torch.no_grad():
            pred0, _ = self.fbonus0(in_)
            pred1, _ = self.fbonus1(in_)
            true0, _ = self.ftrue0(in_)
            true1, _ = self.ftrue1(in_)
        b, _ = torch.max(torch.concat([torch.abs(pred0 - true0), torch.abs(pred1 - true1)], dim=1),
                         dim=1, keepdim=True)
        return b.detach()

    def eval_error_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_q):
        q, _ = self.get_q_value(state_batch, action_batch, with_grad=False)
        target = reward_batch + mask_batch * self.gamma * next_q
        error = nn.functional.mse_loss(q.detach(), target.detach())
        return error

    def eval_error_proposal(self, state_batch, action_batch, network):
        with torch.no_grad():
            logp, _ = network.log_prob(state_batch, action_batch)
        return -logp.mean().detach()

    def eval_error_actor(self, state_batch, action_batch, network):
        with torch.no_grad():
            logp, _ = network.log_prob(state_batch, action_batch)
        return -logp.mean().detach()

    # def eval_error_actor(self, state_batch):
    #     act, _, _ = self.get_policy(state_batch, with_grad=False)
    #     q, _ = self.get_q_value(state_batch, act, with_grad=False)
    #     return -q.mean().detach(), act.detach()

    def eval_value_actor(self, state_batch):
        act, _, _ = self.get_policy(state_batch, with_grad=False)
        q, _ = self.get_q_value(state_batch, act, with_grad=False)
        return q.detach().numpy().mean(), act.detach()

    def parameter_backup_critic(self):
        clone_model_0to1(self.critic, self.critic_copy)
        clone_model_0to1(self.critic_optimizer, self.critic_opt_copy)
        self.critic_lr_weight_copy = self.critic_lr_weight

    def parameter_backup_sampler(self):
        clone_model_0to1(self.sampler, self.sampler_copy)
        clone_model_0to1(self.sampler_optim, self.sampler_opt_copy)
        self.sampler_lr_weight_copy = self.sampler_lr_weight

    def parameter_backup_actor(self):
        clone_model_0to1(self.actor, self.actor_copy)
        clone_model_0to1(self.actor_optimizer, self.actor_opt_copy)
        self.actor_lr_weight_copy = self.actor_lr_weight

    def parameter_backup_explore(self):
        clone_model_0to1(self.fbonus0, self.fbonus0_copy)
        clone_model_0to1(self.fbonus1, self.fbonus1_copy)
        clone_model_0to1(self.bonus_opt_0, self.bonus_opt_0_copy)
        clone_model_0to1(self.bonus_opt_1, self.bonus_opt_1_copy)
        self.explore_lr_weight_copy = self.explore_lr_weight

    def undo_update_critic(self):
        clone_model_0to1(self.critic_copy, self.critic)
        clone_model_0to1(self.critic_opt_copy, self.critic_optimizer)

    def undo_update_sampler(self):
        clone_model_0to1(self.sampler_copy, self.sampler)
        clone_model_0to1(self.sampler_opt_copy, self.sampler_optim)

    def undo_update_actor(self):
        clone_model_0to1(self.actor_copy, self.actor)
        clone_model_0to1(self.actor_opt_copy, self.actor_optimizer)

    def undo_update_explore(self):
        clone_model_0to1(self.fbonus0_copy, self.fbonus0)
        clone_model_0to1(self.fbonus1_copy, self.fbonus1)
        clone_model_0to1(self.bonus_opt_0_copy, self.bonus_opt_0)
        clone_model_0to1(self.bonus_opt_1_copy, self.bonus_opt_1)

    def reset_critic(self):
        if self.cfg.discrete_control:
            self.critic = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim, self.cfg.hidden_critic, self.action_dim,
                                              self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm, self.cfg.critic_ensemble)
            self.critic_target = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim, self.cfg.hidden_critic, self.action_dim,
                                                     self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm, self.cfg.critic_ensemble)
        else:
            self.critic = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim + self.action_dim, self.cfg.hidden_critic, 1,
                                              self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm, self.cfg.critic_ensemble)
            self.critic_target = init_critic_network(self.cfg.critic, self.cfg.device, self.state_dim + self.action_dim, self.cfg.hidden_critic, 1,
                                                     self.cfg.activation, self.cfg.layer_init_critic, self.cfg.layer_norm, self.cfg.critic_ensemble)

        clone_model_0to1(self.critic, self.critic_target)
        self.critic_optimizer = init_optimizer(self.cfg.optimizer, list(self.critic.parameters()), self.cfg.lr_critic)
        self.critic_lr_weight = 1
        self.critic_lr_weight_copy = 1

        for _ in range(self.reset_iteration(self.critic_lr_start, self.cfg.lr_critic)):
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                data['obs2'], 1 - data['done']
            q_loss, _ = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            # do not change lr_weight here
            self.critic_optimizer.zero_grad()
            self.ensemble_critic_loss_backward(q_loss) #q_loss.backward()
            self.critic_optimizer.step()

    def reset_actor(self):
        self.actor = init_policy_network(self.cfg.actor, self.cfg.device, self.state_dim, self.cfg.hidden_actor,
                                         self.action_dim, self.cfg.beta_parameter_bias, self.cfg.beta_parameter_bound, self.cfg.activation,
                                         self.cfg.head_activation, self.cfg.layer_init_actor, self.cfg.layer_norm)
        self.actor_optimizer = init_optimizer(self.cfg.optimizer, list(self.actor.parameters()),
                                              self.cfg.lr_actor)

        self.actor_lr_weight = 1
        self.actor_lr_weight_copy = 1

        for _ in range(self.reset_iteration(self.actor_lr_start, self.cfg.lr_actor)):
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                data['obs2'], 1 - data['done']

            _, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, logp = self.actor_loss(state_batch)
            pi_loss = -logp.mean()
            # do not change lr_weight here
            self.actor_optimizer.zero_grad()
            pi_loss.backward()
            self.actor_optimizer.step()

    def reset_sampler(self):
        self.sampler = init_policy_network(self.cfg.actor, self.cfg.device, self.state_dim, self.cfg.hidden_actor,
                                           self.action_dim,
                                           self.cfg.beta_parameter_bias, self.cfg.beta_parameter_bound, self.cfg.activation,
                                           self.cfg.head_activation, self.cfg.layer_init_actor, self.cfg.layer_norm)
        self.sampler_optim = init_optimizer(self.cfg.optimizer, list(self.sampler.parameters()),
                                            self.lr_sampler)

        for _ in range(self.reset_iteration(self.actor_lr_start, self.cfg.lr_actor)):
            data = self.get_data()
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                data['obs2'], 1 - data['done']
            batch_size = len(state_batch)

            repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)
            with torch.no_grad():
                sample_actions, _, _ = self.sampler(repeated_states)
            q_values, _ = self.get_q_value(repeated_states, sample_actions, with_grad=False)
            # q_values = q_values.reshape(self.batch_size, self.num_samples, 1)
            q_values = q_values.reshape(batch_size, self.num_samples, 1)
            sorted_q = torch.argsort(q_values, dim=1, descending=True)
            best_ind = sorted_q[:, :self.top_action]
            best_ind = best_ind.repeat_interleave(self.gac_a_dim, -1)
            # sample_actions = sample_actions.reshape(self.batch_size, self.num_samples, self.gac_a_dim)
            sample_actions = sample_actions.reshape(batch_size, self.num_samples, self.gac_a_dim)
            best_actions = torch.gather(sample_actions, 1, best_ind)

            # Reshape samples for calculating the loss
            stacked_s_batch = state_batch.repeat_interleave(self.top_action, dim=0)
            best_actions = torch.reshape(best_actions, (-1, self.gac_a_dim))

            sampler_loss = self.proposal_loss(sample_actions, repeated_states,
                                              stacked_s_batch, best_actions, sorted_q, state_batch)
            self.sampler_optim.zero_grad()
            sampler_loss.backward()
            self.sampler_optim.step()

    def backtrack_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                         eval_state, eval_action, eval_reward, eval_next_state, eval_mask):
        next_action, _, _ = self.get_policy(next_state_batch, with_grad=False)
        next_q, _ = self.get_q_value_target(next_state_batch, next_action)

        if self.separated_testset:
            eval_next_action, _, _ = self.get_policy(eval_next_state, with_grad=False)
            eval_next_q, _ = self.get_q_value_target(eval_next_state, eval_next_action)
            before_error = self.eval_error_critic(eval_state, eval_action, eval_reward, eval_mask, eval_next_q)
        else:
            eval_next_q = None
            before_error = self.eval_error_critic(state_batch, action_batch, reward_batch, mask_batch, next_q)

        q_loss, next_action = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        q_loss_weighted = q_loss * self.critic_lr_weight # The weight is supposed to always be 1.
        self.critic_optimizer.zero_grad()
        self.ensemble_critic_loss_backward(q_loss_weighted) #q_loss_weighted.backward()
        grad_rec_critic = clone_gradient(self.critic)

        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                self.critic_optimizer.zero_grad()
                move_gradient_to_network(self.critic, grad_rec_critic, self.critic_lr_weight)
            self.critic_optimizer.step()

            if self.separated_testset:
                after_error = self.eval_error_critic(eval_state, eval_action, eval_reward, eval_mask, eval_next_q)
            else:
                after_error = self.eval_error_critic(state_batch, action_batch, reward_batch, mask_batch, next_q)

            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.critic_lr_weight *= 0.5
                self.undo_update_critic()
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                self.cfg.lr_critic = max(self.cfg.lr_critic * 0.5, self.critic_lr_lower_bound)
                self.critic_optimizer = init_optimizer(self.cfg.optimizer, list(self.critic.parameters()),
                                                       self.cfg.lr_critic)
                # self.reset_critic()
                # print("Critic Done backtracking and hit the limit. Scaler is", self.critic_lr_weight)
                break
            else:
                # print("Critic Done backtracking. Scaler is", self.critic_lr_weight)
                break
        self.last_critic_scaler = self.critic_lr_weight
        self.critic_lr_weight = self.critic_lr_weight_copy
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
        print("sortqvalue")
        print(q_values.mean(), q_values.std(), q_values.min(), q_values.max())
        print(exp_b.mean(), exp_b.std(), exp_b.min(), exp_b.max())
        print("---")
        self.last_explore_bonus = [exp_b.mean(), exp_b.min(), exp_b.max()]
        q_values += self.explore_scaler * exp_b

        q_values = q_values.reshape(batch_size, self.num_samples, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        return sorted_q

    def backtrack_actor(self, state_batch, eval_state):
        pi_loss, repeated_states, sample_actions, sorted_q, stacked_s_batch, best_actions, logp = self.actor_loss(state_batch)

        if self.separated_testset:
            eval_stacked_s, eval_best_action, eval_sample_actions, sorted_eval_q = self.get_best_action_proposal(eval_state)
            before_error = self.eval_error_actor(eval_stacked_s, eval_best_action, self.actor_copy)
        else:
            eval_stacked_s, eval_best_action = None, None
            eval_sample_actions, sorted_eval_q = None, None
            before_error = self.eval_error_actor(stacked_s_batch, best_actions, self.actor_copy)

        pi_loss_weighted = pi_loss
        self.actor_optimizer.zero_grad()
        pi_loss_weighted.backward()

        grad_rec_actor = clone_gradient(self.actor)
        for bi in range(self.max_backtracking):
            if bi > 0:
                self.actor_optimizer.zero_grad()
                move_gradient_to_network(self.actor, grad_rec_actor, self.actor_lr_weight)
            self.actor_optimizer.step()

            if self.separated_testset:
                after_error = self.eval_error_actor(eval_stacked_s, eval_best_action, self.actor)
            else:
                after_error = self.eval_error_actor(stacked_s_batch, best_actions, self.actor)

            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.actor_lr_weight *= 0.5
                self.undo_update_actor()
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                # print("Actor Done backtracking and hit the limit. Scaler is", self.actor_lr_weight)
                self.cfg.lr_actor = max(self.cfg.lr_actor * 0.5, self.actor_lr_lower_bound)
                self.actor_optimizer = init_optimizer(self.cfg.optimizer, list(self.actor.parameters()),
                                                      self.cfg.lr_actor)
                # self.reset_actor()
                break
            else:
                # print("Actor Done backtracking. Scaler is", self.actor_lr_weight)
                break
        self.last_actor_scaler = self.actor_lr_weight
        self.actor_lr_weight = self.actor_lr_weight_copy
        return (sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                eval_sample_actions, sorted_eval_q)

    def backtrack_sampler(self, sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                          eval_state, eval_sample_actions, sorted_eval_q):
        sampler_loss = self.proposal_loss(sample_actions, repeated_states,
                                          stacked_s_batch, best_actions, sorted_q, state_batch)
        if self.separated_testset:
            eval_stacked_s, eval_best_action = self.get_action_with_top_value(eval_state, eval_sample_actions, sorted_eval_q,
                                                                              self.top_action_proposal)
            before_error = self.eval_error_proposal(eval_stacked_s, eval_best_action, self.sampler_copy)
        else:
            eval_stacked_s, eval_best_action = None, None
            before_error = self.eval_error_proposal(stacked_s_batch, best_actions, self.sampler_copy)

        self.sampler_optim.zero_grad()
        sampler_loss.backward()

        grad_rec_proposal = clone_gradient(self.sampler)
        for bi in range(self.max_backtracking):
            if bi > 0:
                self.sampler_optim.zero_grad()
                move_gradient_to_network(self.sampler, grad_rec_proposal, self.sampler_lr_weight)
            self.sampler_optim.step()

            if self.separated_testset:
                after_error = self.eval_error_proposal(eval_stacked_s, eval_best_action, self.sampler)
            else:
                after_error = self.eval_error_proposal(stacked_s_batch, best_actions, self.sampler)

            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.sampler_lr_weight *= 0.5
                self.undo_update_sampler()
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                # print("Sampler Done backtracking and hit the limit. Scaler is", self.critic_lr_weight)
                self.lr_sampler = max(self.lr_sampler * 0.5, self.actor_lr_lower_bound)
                self.sampler_optim = init_optimizer(self.cfg.optimizer, list(self.sampler.parameters()),
                                                    self.lr_sampler)
                # self.reset_sampler()
                break
            else:
                break
        self.last_sampler_scaler = self.sampler_lr_weight if bi < self.max_backtracking-1 else 0
        self.sampler_lr_weight = self.sampler_lr_weight_copy
        return


    def reset_iteration(self, lr_start, lr_current):
        count = (int(np.ceil(self.buffer.size / self.cfg.batch_size)) *
                 (int(np.log2(lr_start / lr_current)))) * 5
        return count


    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (data['obs'], data['act'], data['reward'],
                                                                                 data['obs2'], 1. - data['done'])

        eval_data = self.get_all_data()
        eval_state, eval_action, eval_reward, eval_next_state, eval_mask = (eval_data['obs'], eval_data['act'], eval_data['reward'],
                                                                            eval_data['obs2'], 1. - eval_data['done'])

        # critic update
        self.parameter_backup_critic()
        next_action = self.backtrack_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                                            eval_state, eval_action, eval_reward, eval_next_state, eval_mask)

        # uncertainty update
        self.parameter_backup_explore()
        self.explore_bonus_update(state_batch, action_batch, reward_batch, next_state_batch, next_action, mask_batch,
                                  eval_state, eval_action, eval_reward, eval_next_state, eval_mask)

        # actor update
        self.parameter_backup_actor()
        sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch, eval_sample_actions, sorted_eval_q = \
            self.backtrack_actor(state_batch, eval_state)

        # proposal update
        self.parameter_backup_sampler()
        self.backtrack_sampler(sample_actions, repeated_states, stacked_s_batch, best_actions, sorted_q, state_batch,
                               eval_state, eval_sample_actions, sorted_eval_q)

    def agent_debug_info(self, observation_tensor, action_tensor, pi_info, env_info):
        i_log = super(LineSearchAgent, self).agent_debug_info(observation_tensor, action_tensor, pi_info, env_info)
        i_log["lr_actor"] = self.cfg.lr_actor
        i_log["lr_critic"] = self.cfg.lr_critic
        i_log["lr_sampler"] = self.lr_sampler
        i_log["lr_actor_scaler"] = self.last_actor_scaler
        i_log["lr_sampler_scaler"] = self.last_sampler_scaler
        i_log["lr_critic_scaler"] = self.last_critic_scaler
        i_log["lr_explore_scaler"] = self.last_explore_scaler
        i_log["explore_bonus"] = self.last_explore_bonus
        return i_log


class LineSearchBU(LineSearchAgent):
    """
    Batch Update
    """
    def __init__(self, cfg, average_entropy=True):
        super(LineSearchBU, self).__init__(cfg, average_entropy)
        self.separated_testset = False

    def get_data(self):
        return self.get_all_data()


class LineSearchVB(LineSearchAgent):
    """
    Value based method
    """
    def __init__(self, cfg, average_entropy=True):
        super(LineSearchVB, self).__init__(cfg, average_entropy)
        self.exploration = cfg.exploration
        del self.actor_copy, self.sampler_copy, self.actor, self.sampler
        self.random_sampler = init_policy_network("Beta", self.cfg.device, self.state_dim, [], self.action_dim,
                                                  0, 0, self.cfg.activation, self.cfg.head_activation, "Const/1/0", False)
        self.actor = init_policy_network(self.cfg.actor, self.cfg.device, self.state_dim, self.cfg.hidden_actor,
                                         self.action_dim, self.cfg.beta_parameter_bias, self.cfg.beta_parameter_bound, self.cfg.activation,
                                         self.cfg.head_activation, self.cfg.layer_init_actor, self.cfg.layer_norm)
        self.actor_optimizer = init_optimizer(self.cfg.optimizer, list(self.actor.parameters()),
                                              self.cfg.lr_actor)

    def inner_update(self, trunc=False):
        data = self.get_data()
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], \
                                                                                data['obs2'], 1 - data['done']
        # critic update
        q_loss, _ = self.critic_loss(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.critic_optimizer.zero_grad()
        self.ensemble_critic_loss_backward(q_loss) #q_loss.backward()
        self.critic_optimizer.step()
        return data

    def greedy_sampling(self, states):
        """Take the max form random samples"""
        rept_states = states.repeat_interleave(self.num_samples, dim=0)
        with torch.no_grad():
            sample_actions, _, _ = self.random_sampler(rept_states)
        qs, _ = self.get_q_value(rept_states, sample_actions, with_grad=False)
        sorted_qs = torch.argsort(qs, dim=1, descending=True)
        _, best_action = self.get_action_with_top_value(states, sample_actions, sorted_qs, 1)
        return best_action

    def get_policy(self, observation, with_grad=False, debug=False):
        if self.rng.random() >= self.exploration:
            a = self.greedy_sampling(observation)
        else:
            a, _, _ = self.random_sampler(observation)
        return a, None, None
